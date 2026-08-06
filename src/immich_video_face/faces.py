"""Face detection and embedding.

Deliberately mirrors what Immich's ML container does, because our embeddings
have to land in the same vector space as the ones stored in `face_search`:

    detection  -> RetinaFace (buffalo_l det_10g), giving boxes + 5 landmarks
    alignment  -> insightface.utils.face_align.norm_crop(image, landmarks)
    recognition-> ArcFace, L2-normalised

InsightFace's `FaceAnalysis` performs exactly this sequence internally, so we
use it rather than re-implementing the steps. Images are BGR uint8 throughout,
matching Immich's `decode_cv2`. Verified by the `parity` command.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from .config import DetectSettings

log = logging.getLogger(__name__)


@dataclass
class Observation:
    """One detected face in one frame."""

    frame_index: int
    # (x1, y1, x2, y2) in the *scaled* frame's coordinate space.
    box: tuple[float, float, float, float]
    det_score: float
    embedding: np.ndarray  # L2-normalised, 512-d
    blur: float  # Laplacian variance; higher is sharper
    # Populated only when reports are requested. Holding the ~30KB face crop
    # here rather than the whole frame keeps a long backfill from ballooning:
    # retaining every frame that contained a face would run to gigabytes.
    crop: np.ndarray | None = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def quality(self) -> float:
        """Ranking score for picking a cluster's representative frame."""
        return self.det_score * np.sqrt(max(self.blur, 1.0)) * np.sqrt(max(self.area, 1.0))


def preload_cuda_libraries() -> None:
    """Make pip-installed NVIDIA runtime libraries visible to onnxruntime.

    The CUDA execution provider links against libcublas/libcudart/etc., which
    the `cuda` extra installs under site-packages/nvidia/*/lib. The dynamic
    loader will not look there on its own and LD_LIBRARY_PATH cannot be changed
    usefully after the process starts, so we load them globally by hand.
    Without this onnxruntime silently falls back to CPU.
    """
    import ctypes
    import sysconfig
    from pathlib import Path

    site_packages = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not site_packages.is_dir():
        return
    for lib_dir in sorted(site_packages.glob("*/lib")):
        for so in sorted(lib_dir.glob("*.so*")):
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                # Many of these have interdependencies; a failure here is only
                # fatal if the provider itself then fails to load.
                continue


class FaceDetector:
    def __init__(self, settings: DetectSettings) -> None:
        self.settings = settings
        self._app = self._build()

    def _build(self):
        if self.settings.device == "cuda":
            preload_cuda_libraries()

        # Imported lazily so that config/DB-only commands work without the
        # (large, GPU-bound) ML stack installed.
        from insightface.app import FaceAnalysis

        if self.settings.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        root = os.environ.get("INSIGHTFACE_ROOT", os.path.expanduser("~/.insightface"))
        app = FaceAnalysis(
            name=self.settings.model,
            root=root,
            # Only detection and recognition; skip age/gender/landmark extras.
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        app.prepare(
            ctx_id=0 if self.settings.device == "cuda" else -1,
            det_size=(self.settings.det_size, self.settings.det_size),
            det_thresh=self.settings.min_score,
        )

        self.active_provider = _active_provider(app)
        if self.settings.device == "cuda" and self.active_provider != "CUDAExecutionProvider":
            # Worth shouting about: a silent CPU fallback is ~20x slower and is
            # easy to miss in a long backfill.
            log.warning(
                "Requested CUDA but running on %s. Install the 'cuda' extra "
                "(uv pip install -e '.[cuda]').",
                self.active_provider,
            )
        else:
            log.info("Loaded %s on %s", self.settings.model, self.active_provider)
        return app

    def detect(
        self, frame: np.ndarray, frame_index: int = 0, keep_crop: bool = False
    ) -> list[Observation]:
        """Detect and embed every face in a single BGR frame."""
        faces = self._app.get(frame)
        observations: list[Observation] = []
        for face in faces:
            score = float(face.det_score)
            if score < self.settings.min_score:
                continue
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                continue
            box = tuple(float(v) for v in face.bbox)
            if max(box[2] - box[0], box[3] - box[1]) < self.settings.min_face_px:
                continue
            patch = crop(frame, box)
            observations.append(
                Observation(
                    frame_index=frame_index,
                    box=box,  # type: ignore[arg-type]
                    det_score=score,
                    embedding=np.asarray(embedding, dtype=np.float32),
                    blur=laplacian_variance(patch),
                    # `frame` is a view over ffmpeg's reused buffer, so copy.
                    crop=patch.copy() if keep_crop else None,
                )
            )
        return observations


def _active_provider(app) -> str:
    """The provider onnxruntime actually chose, not the one we asked for."""
    for model in getattr(app, "models", {}).values():
        session = getattr(model, "session", None)
        if session is not None:
            providers = session.get_providers()
            if providers:
                return providers[0]
    return "unknown"


def crop(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(width, int(box[2]))
    y2 = min(height, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return frame[:1, :1]
    return frame[y1:y2, x1:x2]


def laplacian_variance(patch: np.ndarray) -> float:
    """Sharpness proxy. Motion-blurred video frames score low and get
    down-weighted when the cluster centroid is computed."""
    import cv2

    if patch.size == 0:
        return 0.0
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
