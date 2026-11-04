"""Group per-frame detections into one cluster per person-appearance.

Matching every frame independently would be both noisy and wasteful: a single
blurry frame can pull a false identity, and a 30-second clip yields hundreds of
near-duplicate embeddings. Clustering first means we match a *centroid of many
observations*, which is markedly more robust, and it cuts matching work by
roughly two orders of magnitude.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from .config import ClusterSettings
from .faces import Observation
from .immich_db import normalise

log = logging.getLogger(__name__)

# Agglomerative clustering is O(n^2) in memory; beyond this we subsample.
MAX_OBSERVATIONS = 6000


@dataclass
class FaceCluster:
    observations: list[Observation]
    centroid: np.ndarray  # L2-normalised
    # When this face is on screen: [(start_s, end_s), ...], filled in by the
    # scanner once frame timestamps are known.
    segments: list[tuple[float, float]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.observations)

    @property
    def best(self) -> Observation:
        """Sharpest, largest, highest-scoring observation -- the one whose
        bounding box we write back and whose crop appears in reports."""
        return max(self.observations, key=lambda o: o.quality)

    def frame_range(self) -> tuple[int, int]:
        indices = [o.frame_index for o in self.observations]
        return min(indices), max(indices)

    def time_range(self, frames) -> tuple[float, float]:
        start, end = self.frame_range()
        return frames.timestamp_for(start), frames.timestamp_for(end)

    def time_segments(self, frames, gap_s: float, pad_s: float) -> list[tuple[float, float]]:
        times = [frames.timestamp_for(o.frame_index) for o in self.observations]
        return time_segments(times, gap_s=gap_s, pad_s=pad_s)


def time_segments(
    times: list[float], *, gap_s: float, pad_s: float, duration_s: float | None = None
) -> list[tuple[float, float]]:
    """Turn the timestamps a face was seen at into on-screen intervals.

    A face is sampled, not tracked, so each sighting stands for roughly one
    sampling interval: `pad_s` (half an interval) is added either side. A gap
    longer than `gap_s` between sightings ends one appearance and starts the next.
    """
    clean = sorted(t for t in times if not math.isnan(t))
    if not clean:
        return []
    segments: list[list[float]] = [[clean[0], clean[0]]]
    for t in clean[1:]:
        if t - segments[-1][1] > gap_s:
            segments.append([t, t])
        else:
            segments[-1][1] = t
    out = []
    for start, end in segments:
        start = max(0.0, start - pad_s)
        end = end + pad_s
        if duration_s:
            end = min(end, duration_s)
        out.append((round(start, 3), round(end, 3)))
    return merge_segments(out)


def merge_segments(
    segments: list[tuple[float, float]], gap_s: float = 0.0
) -> list[tuple[float, float]]:
    """Union of intervals, sorted. Overlapping ones, and ones separated by no more
    than `gap_s`, become one."""
    merged: list[list[float]] = []
    for start, end in sorted(segments):
        if merged and start <= merged[-1][1] + gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def cluster_observations(
    observations: list[Observation], settings: ClusterSettings
) -> list[FaceCluster]:
    """Cluster within a single video, then drop clusters that are too small."""
    if not observations:
        return []

    working = observations
    if len(working) > MAX_OBSERVATIONS:
        log.warning(
            "Subsampling %d observations to %d for clustering",
            len(working),
            MAX_OBSERVATIONS,
        )
        step = len(working) / MAX_OBSERVATIONS
        working = [working[int(i * step)] for i in range(MAX_OBSERVATIONS)]

    if len(working) == 1:
        labels = np.array([0])
    else:
        from sklearn.cluster import AgglomerativeClustering

        matrix = normalise(np.vstack([o.embedding for o in working]))
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=settings.distance,
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(matrix)

    clusters: list[FaceCluster] = []
    for label in sorted(set(labels.tolist())):
        members = [obs for obs, lab in zip(working, labels) if lab == label]
        # A face seen in only one or two frames is usually a detector artefact.
        if len(members) < settings.min_observations:
            continue
        clusters.append(FaceCluster(observations=members, centroid=_centroid(members)))

    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


def _centroid(observations: list[Observation]) -> np.ndarray:
    """Blur-weighted mean embedding.

    Sharp, confidently-detected frames should dominate; motion-smeared ones
    still contribute but carry less weight.
    """
    matrix = np.vstack([o.embedding for o in observations])
    weights = np.array(
        [o.det_score * np.log1p(max(o.blur, 0.0)) for o in observations], dtype=np.float32
    )
    if not np.any(weights > 0):
        weights = np.ones(len(observations), dtype=np.float32)
    mean = np.average(matrix, axis=0, weights=weights)
    return normalise(mean)[0]
