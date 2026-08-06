"""Group per-frame detections into one cluster per person-appearance.

Matching every frame independently would be both noisy and wasteful: a single
blurry frame can pull a false identity, and a 30-second clip yields hundreds of
near-duplicate embeddings. Clustering first means we match a *centroid of many
observations*, which is markedly more robust, and it cuts matching work by
roughly two orders of magnitude.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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
