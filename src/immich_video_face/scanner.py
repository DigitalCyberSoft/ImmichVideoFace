"""Per-video scan pipeline: decode -> detect -> cluster -> match."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cluster import FaceCluster, cluster_observations  # noqa: F401  (re-exported)
from .config import Config
from .faces import FaceDetector, Observation
from .immich_api import VideoAsset
from .match import Matcher, MatchResult
from .video import FrameBatch, VideoError, iter_frames, probe

log = logging.getLogger(__name__)


@dataclass
class ScanResult:
    asset: VideoAsset
    path: Path
    frames: int = 0
    observations: int = 0
    clusters: list[FaceCluster] = field(default_factory=list)
    matches: list[MatchResult] = field(default_factory=list)
    # Detection space; what we pass to Immich as imageWidth/imageHeight so it
    # can scale the bounding box itself.
    frame_width: int = 0
    frame_height: int = 0
    error: str | None = None
    # Best observation crop per cluster, kept only when reports are requested.
    crops: dict[int, np.ndarray] = field(default_factory=dict)
    duration_s: float = 0.0
    # (timestamp_s, CLIP embedding) for the frames sampled for object search.
    frame_embeddings: list[tuple[float, np.ndarray]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def matched(self) -> list[MatchResult]:
        return [m for m in self.matches if m.matched]

    @property
    def unmatched(self) -> list[MatchResult]:
        return [m for m in self.matches if not m.matched]


class Scanner:
    def __init__(self, config: Config, detector: FaceDetector, matcher: Matcher) -> None:
        self.config = config
        self.detector = detector
        self.matcher = matcher

    def scan(self, asset: VideoAsset, keep_crops: bool = False) -> ScanResult:
        path = self.config.paths.to_host(asset.original_path)
        result = ScanResult(asset=asset, path=path)

        if not path.exists():
            result.error = f"file not found: {path}"
            return result

        try:
            info = probe(path)
            frames_iter, batch = iter_frames(path, self.config.sampling, info)
        except VideoError as exc:
            result.error = str(exc)
            return result

        result.frame_width = batch.width
        result.frame_height = batch.height
        result.duration_s = info.duration_s

        # Object search: every `clip_every`-th sampled frame also gets a CLIP
        # embedding. Not for a Live Photo's motion clip -- it is hidden, so search
        # could never return it, and the still beside it has its own embedding.
        clip_every = 0
        if self.detector.clip_model and not asset.still_asset_id:
            fps = max(self.config.sampling.min_fps, 0.1)
            clip_every = max(1, round(self.config.objects.interval_s * fps))

        observations: list[Observation] = []
        clip_vectors: list[tuple[int, np.ndarray]] = []

        def collect(index: int, outcome) -> None:
            found, clip_vector = outcome
            observations.extend(found)
            if clip_vector is not None:
                clip_vectors.append((index, clip_vector))

        def wants_clip(index: int) -> bool:
            return bool(clip_every) and index % clip_every == 0

        workers = (
            max(1, self.config.detect.ml_workers)
            if self.config.detect.backend == "immich-ml"
            else 1
        )
        try:
            if workers == 1:
                for index, frame in frames_iter:
                    result.frames += 1
                    collect(
                        index,
                        self.detector.analyse(frame, index, keep_crops, wants_clip(index)),
                    )
            else:
                # Keep several frames in flight so the ML service can work on them
                # in parallel; results are collected in frame order. Each frame is
                # its own buffer (ffmpeg output is read into fresh bytes), so it is
                # safe to hand to another thread.
                from collections import deque
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    inflight: deque = deque()
                    for index, frame in frames_iter:
                        result.frames += 1
                        inflight.append(
                            (
                                index,
                                pool.submit(
                                    self.detector.analyse,
                                    frame,
                                    index,
                                    keep_crops,
                                    wants_clip(index),
                                ),
                            )
                        )
                        if len(inflight) >= workers * 2:
                            done_index, future = inflight.popleft()
                            collect(done_index, future.result())
                    while inflight:
                        done_index, future = inflight.popleft()
                        collect(done_index, future.result())
        except Exception as exc:  # pragma: no cover - decode failures vary widely
            result.error = f"decode/detect failed: {exc}"
            return result

        result.observations = len(observations)
        result.clusters = cluster_observations(observations, self.config.cluster)

        # Frame timestamps are complete now that ffmpeg has exited.
        pad_s = 0.5 / max(self.config.sampling.min_fps, 0.1)
        for cluster in result.clusters:
            cluster.segments = cluster.time_segments(
                batch, gap_s=self.config.timeline.gap_s, pad_s=pad_s
            )
        result.frame_embeddings = [
            (batch.timestamp_for(index), vector)
            for index, vector in clip_vectors
            if not np.isnan(batch.timestamp_for(index))
        ]

        result.matches = self.matcher.match_all(result.clusters)
        # A person split over several clusters has their screen time merged; close
        # the small gaps that leaves where one cluster hands over to the next.
        from .cluster import merge_segments

        bridge = max(0.0, self.config.timeline.gap_s - 2 * pad_s)
        for match in result.matches:
            match.segments = merge_segments(match.segments, gap_s=bridge)

        if keep_crops:
            result.crops = {
                index: cluster.best.crop
                for index, cluster in enumerate(result.clusters)
                if cluster.best.crop is not None
            }

        _log_summary(result, batch)
        return result


def _log_summary(result: ScanResult, batch: FrameBatch) -> None:
    names = ", ".join(
        f"{m.person_name or m.person_id[:8]}({m.distance:.3f})" for m in result.matched
    )
    log.info(
        "%s: %d frames, %d faces, %d clusters -> %s",
        result.path.name,
        result.frames,
        result.observations,
        len(result.clusters),
        names or "no match",
    )
