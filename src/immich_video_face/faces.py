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
    def __init__(self, settings: DetectSettings, clip_model: str | None = None) -> None:
        self.settings = settings
        self.clip_model = clip_model
        self._ml: ImmichMlClient | None = None
        if clip_model and settings.backend != "immich-ml":
            raise ValueError("Object search needs detect.backend = \"immich-ml\"")
        if settings.backend == "immich-ml":
            self._ml = ImmichMlClient(settings)
            self._app = None
            self.active_provider = f"immich-ml ({settings.ml_url})"
            log.info("Using Immich machine learning at %s, model %s", settings.ml_url, settings.model)
        else:
            self._app = self._build()

    def close(self) -> None:
        if self._ml is not None:
            self._ml.close()

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
        return self.analyse(frame, frame_index, keep_crop)[0]

    def analyse(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        keep_crop: bool = False,
        clip: bool = False,
    ) -> tuple[list[Observation], np.ndarray | None]:
        """Faces in a frame and, when `clip` is set, the frame's CLIP embedding.

        Both come from a single request to the ML service, so object search adds
        no extra decoding and only one more model pass per sampled frame.
        """
        clip_vector = None
        if self._ml is not None:
            faces, clip_vector = self._ml.predict(
                frame, clip_model=self.clip_model if clip else None
            )
        else:
            faces = [
                (tuple(float(v) for v in face.bbox), float(face.det_score),
                 getattr(face, "normed_embedding", None))
                for face in self._app.get(frame)
            ]
        observations: list[Observation] = []
        for box, score, embedding in faces:
            if score < self.settings.min_score:
                continue
            if embedding is None:
                continue
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
        return observations, clip_vector


class ImmichMlClient:
    """Detection and recognition through Immich's machine-learning service.

    This is the same endpoint Immich's server calls for photos
    (`POST /predict` with an `entries` JSON and an `image` file), so the boxes,
    scores and embeddings are produced exactly as Immich's own are. Frames are
    sent as high-quality JPEG; Immich itself runs face detection on JPEG
    previews, so this matches how the stored gallery was made.
    """

    def __init__(self, settings: DetectSettings) -> None:
        import json

        import httpx

        self.settings = settings
        self._json = json
        self._client = httpx.Client(base_url=settings.ml_url.rstrip("/"), timeout=300.0)
        self._faces_entry = {
            "detection": {
                "modelName": settings.model,
                "options": {"minScore": settings.min_score},
            },
            "recognition": {"modelName": settings.model},
        }

    def close(self) -> None:
        self._client.close()

    def faces(
        self, frame: np.ndarray
    ) -> list[tuple[tuple[float, float, float, float], float, np.ndarray]]:
        return self.predict(frame)[0]

    def predict(
        self, frame: np.ndarray, clip_model: str | None = None
    ) -> tuple[list[tuple[tuple[float, float, float, float], float, np.ndarray]], np.ndarray | None]:
        import cv2

        from .immich_db import normalise, parse_vector

        entries: dict = {"facial-recognition": self._faces_entry}
        if clip_model:
            entries["clip"] = {"visual": {"modelName": clip_model}}
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Could not encode frame as JPEG")
        response = self._client.post(
            "/predict",
            data={"entries": self._json.dumps(entries)},
            files={"image": ("frame.jpg", jpeg.tobytes(), "image/jpeg")},
        )
        response.raise_for_status()
        payload = response.json()
        clip_vector = None
        if clip_model and payload.get("clip") is not None:
            # Stored as Immich stores smart search vectors: as returned (unit length).
            clip_vector = parse_vector(payload["clip"])
        results = []
        for face in payload.get("facial-recognition") or []:
            b = face["boundingBox"]
            box = (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]))
            # Immich stores raw ArcFace vectors; everything here works on unit vectors.
            embedding = normalise(parse_vector(face["embedding"]))[0]
            results.append((box, float(face["score"]), embedding))
        return results, clip_vector


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
