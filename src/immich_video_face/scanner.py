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

        observations: list[Observation] = []
        try:
            for index, frame in frames_iter:
                result.frames += 1
                observations.extend(
                    self.detector.detect(frame, frame_index=index, keep_crop=keep_crops)
                )
        except Exception as exc:  # pragma: no cover - decode failures vary widely
            result.error = f"decode/detect failed: {exc}"
            return result

        result.observations = len(observations)
        result.clusters = cluster_observations(observations, self.config.cluster)
        result.matches = self.matcher.match_all(result.clusters)

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
