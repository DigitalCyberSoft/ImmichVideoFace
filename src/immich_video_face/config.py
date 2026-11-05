"""Configuration loading and the fingerprint that drives re-scan decisions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tomllib
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DEFAULT_CONFIG_PATHS = (
    Path("config.toml"),
    Path.home() / ".config" / "immich-video-face" / "config.toml",
)


class ImmichSettings(BaseModel):
    url: str = "http://localhost:2283"
    api_key: str = ""
    # Left blank, resolved from the API key's own user at startup.
    owner_id: str = ""
    timeout: float = 30.0
    # Address for links in reports. `url` is usually localhost, which is right
    # for the API but useless in a browser on another machine.
    public_url: str = ""

    @property
    def browse_url(self) -> str:
        return self.public_url or self.url


class PathSettings(BaseModel):
    """Immich stores container-absolute paths; we read the files from the host."""

    data_root_container: str = "/data"
    data_root_host: str = "/var/lib/immich/library"

    def to_host(self, original_path: str) -> Path:
        if original_path.startswith(self.data_root_container):
            suffix = original_path[len(self.data_root_container) :].lstrip("/")
            return Path(self.data_root_host) / suffix
        return Path(original_path)


class DbSettings(BaseModel):
    # "docker:<container>" resolves the container's bridge IP at startup, since
    # Immich does not publish 5432 to the host and the IP moves on restart.
    host: str = "docker:immich_postgres"
    port: int = 5432
    name: str = "immich"
    user: str = "postgres"
    password: str = ""

    def resolve_host(self) -> str:
        if not self.host.startswith("docker:"):
            return self.host
        container = self.host.split(":", 1)[1]
        result = subprocess.run(
            [
                "docker",
                "inspect",
                container,
                "--format",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not resolve container {container!r} via docker inspect: "
                f"{result.stderr.strip()}. Set db.host to an explicit address instead."
            )
        ips = result.stdout.split()
        if not ips:
            raise RuntimeError(f"Container {container!r} has no network address.")
        return ips[0]

    def conninfo(self) -> str:
        """libpq connection string.

        Built from keywords rather than a URL so that `host` may be a Unix socket
        directory (e.g. /run/postgresql) and `password` may be left blank for peer
        authentication -- the natural setup when Immich runs without containers.
        """
        from psycopg.conninfo import make_conninfo

        params = {
            "host": self.resolve_host(),
            "port": self.port,
            "dbname": self.name,
            "user": self.user,
        }
        if self.password:
            params["password"] = self.password
        return make_conninfo(**params)


class SelectSettings(BaseModel):
    """Which videos are worth scanning, and where their results belong."""

    # A Live Photo's motion clip is a hidden asset: a face written there is never
    # displayed, because the still image beside it is what Immich shows.
    #
    #   redirect - scan the clip, write faces onto the paired still (default).
    #              Anyone already detected on the still is skipped as a duplicate,
    #              so only people who appear *in the motion but not the still*
    #              produce a write. That is the gap worth filling.
    #   skip     - ignore these clips entirely.
    #   scan     - write onto the hidden clip itself. Almost always pointless.
    live_photos: Literal["redirect", "skip", "scan"] = "redirect"


class SamplingSettings(BaseModel):
    keyframes: bool = True
    # Brief appearances are the ones that get lost: at 1 fps a three-second
    # appearance yields three observations, which `min_observations` discards if
    # clustering splits them at all. On a GPU decode dominates runtime rather than
    # frame count, so a high rate is cheap there. On CPU every frame costs a full
    # detection pass, so "auto" picks `gpu_fps` or `cpu_fps` by what detection runs on.
    min_fps: float | Literal["auto"] = "auto"
    gpu_fps: float = 8.0
    cpu_fps: float = 0.5
    max_long_edge: int = 960
    # Safety valve for the handful of very long videos in the library.
    max_frames: int = 3000


class DetectSettings(BaseModel):
    # Where detection and embedding run:
    #   insightface - in this process, with InsightFace and onnxruntime (GPU or CPU)
    #   immich-ml   - Immich's own machine-learning service, over HTTP. Embeddings
    #                 then come from the exact code Immich uses for photos, no second
    #                 copy of the models is loaded, and no GPU stack is needed.
    backend: Literal["insightface", "immich-ml"] = "insightface"
    # Whether detection runs on a GPU; drives the "auto" sampling settings.
    # "auto": the insightface backend asks onnxruntime for CUDA; for immich-ml the
    # service picks its own device, so a GPU is assumed only when this host has an
    # NVIDIA one (nvidia-smi). Set true/false when that guess is wrong.
    gpu: bool | Literal["auto"] = "auto"
    # immich-ml only: the service URL, and how many frames to have in flight.
    # The service runs requests in parallel, so a few workers keep every core busy.
    ml_url: str = "http://127.0.0.1:3003"
    ml_workers: int = 2
    model: str = "buffalo_l"
    min_score: float = 0.7
    device: str = "cuda"
    det_size: int = 640
    batch_size: int = 16
    # ArcFace aligns to 112x112, so a face detected much smaller than that is
    # upscaled guesswork. Those embeddings are noisy enough to fragment
    # clustering, and on real footage discarding them cost no matches while
    # cutting cluster count by 2-4x.
    min_face_px: int = 40


class ObjectSettings(BaseModel):
    """Object search across the whole video, not just its thumbnail.

    Every `interval_s` of video, the frame is also sent for a CLIP embedding -- the
    same vector Immich's smart search stores for a photo. They are published to the
    video index (see `IndexSettings`), where a patched Immich ranks a video by its
    best-matching frame. Needs `detect.backend = "immich-ml"`.
    """

    enabled: bool = False
    interval_s: float = 2.0
    # MUST match Immich's smart search model (Administration -> Settings ->
    # Machine Learning -> Smart Search). `ivf check` compares the two.
    model: str = "ViT-B-32__openai"


class TimelineSettings(BaseModel):
    """When each person is on screen."""

    # Someone absent from the sampled frames for longer than this starts a new
    # appearance. A little over the sampling interval bridges a missed detection
    # (a turned head, motion blur) without merging separate appearances.
    # "auto": 3 s, or 2.5 sampling intervals when that is longer.
    gap_s: float | Literal["auto"] = "auto"


class IndexSettings(BaseModel):
    """Publish frame embeddings and appearance times to Postgres.

    Everything goes into one schema that the tool's own database role owns, so
    Immich's tables are still never written. Immich ignores the schema unless it
    carries the matching patch, which reads it for smart search and the
    people-in-this-video timeline.
    """

    enabled: bool = False
    schema_name: str = "ivf"


class SensitiveCategory(BaseModel):
    """One kind of content to flag, as text the search model can compare images with.

    Two cosine similarities, from Immich's own smart search embedding: to the
    nearest of `prompts`, and to the nearest of `contrast` (what is easily
    confused with the category but should stay visible). An image is flagged when
    the first reaches `min_similarity` and leads the second by `min_margin`; a
    video when one sampled frame does. The defaults were measured on public test
    images with ViT-SO400M-16-SigLIP2-384; other models need their own.
    """

    prompts: list[str]
    contrast: list[str]
    min_similarity: float = 0.12
    min_margin: float = 0.0
    # A video needs this many sampled frames to pass (all of them, if it has fewer):
    # one stray frame of a long video is more often a misreading than the real thing.
    min_frames: int = 2


_EVERYDAY = [
    "a photo of a couple", "a portrait of a person", "a group of people", "a family photo",
    "a landscape", "a photo of food", "a child playing", "a selfie",
]
# Everyday scenes that phone video frames (blurred, close up) can resemble.
_LOOKALIKES = [
    "a person eating", "a baby drinking from a bottle", "a baby with a pacifier",
    "a close-up of a face", "a child in the bath", "a baby being changed", "a person in bed",
    "people hugging",
]


def _default_categories() -> dict[str, SensitiveCategory]:
    return {
        "Explicit": SensitiveCategory(
            prompts=[
                "a photo of people having sex", "pornography", "an explicit sexual photo",
                "a photo of exposed genitals", "a photo of oral sex",
                "a photo of a person masturbating",
            ],
            contrast=[
                "a photo of a nude person", "a topless woman at the beach", "a woman in lingerie",
                "a person in a bikini", "a nude statue", "a painting of a nude",
                "a mother breastfeeding a baby", *_EVERYDAY, *_LOOKALIKES,
            ],
            # 95% of explicit test images; 8% of lingerie, nothing else.
            min_similarity=0.119,
            min_margin=-0.01,
        ),
        "Nudity": SensitiveCategory(
            prompts=[
                "a photo of a nude person", "a topless woman", "a photo of exposed breasts",
                "a photo of people having sex", "pornography",
            ],
            contrast=[
                "a woman in lingerie", "a person in a bikini", "a shirtless man",
                "a nude statue", "a painting of a nude", *_EVERYDAY, *_LOOKALIKES,
            ],
            # 98% of nude and 45% of topless test images; 15% of breastfeeding,
            # 8% of lingerie. Pair it with Explicit, which it only partly covers.
            min_similarity=0.11,
            min_margin=0.0,
        ),
        "Underwear": SensitiveCategory(
            prompts=["a woman in lingerie", "a person in underwear", "a woman in a bra"],
            contrast=["a person in a bikini", "a person in a swimsuit", *_EVERYDAY],
            # 68% of lingerie test images; 2% of bikini, 22% of nude.
            min_similarity=0.08,
            min_margin=0.01,
        ),
    }


class SensitiveSettings(BaseModel):
    """Flag sensitive content with a tag, for a patched Immich to hide behind the PIN.

    `ivf sensitive` scores every photo, and every video by its sampled frames,
    against each category in `flag`, and tags what passes `<tag>/<category>`.
    A tag removed by hand is never put back, and a manual tag is never touched.
    """

    enabled: bool = False
    # MUST match the server's IMMICH_SENSITIVE_TAG; anything tagged under it is hidden.
    tag: str = "Sensitive"
    flag: list[str] = Field(default_factory=lambda: ["Explicit"])
    categories: dict[str, SensitiveCategory] = Field(default_factory=_default_categories)
    # Make sure no person's cover photo comes from a sensitive asset.
    fix_cover_faces: bool = True

    @model_validator(mode="before")
    @classmethod
    def _merge_default_categories(cls, data):
        """A category set in the config file changes only the fields it names, so
        `[sensitive.categories.Explicit] min_margin = 0.02` keeps the prompts."""
        if isinstance(data, dict) and data.get("categories"):
            merged = {name: c.model_dump() for name, c in _default_categories().items()}
            for name, override in data["categories"].items():
                merged[name] = {**merged.get(name, {}), **dict(override)}
            data = {**data, "categories": merged}
        return data

    @model_validator(mode="after")
    def _known_flags(self):
        unknown = [name for name in self.flag if name not in self.categories]
        if unknown:
            raise ValueError(
                f"sensitive.flag names unknown categories {unknown}; "
                f"known: {sorted(self.categories)}"
            )
        return self


class ClusterSettings(BaseModel):
    # Cosine distance between two views of the *same* face across a video --
    # different angles, motion blur and lighting -- routinely exceeds 0.4.
    # At 0.35 appearances shattered into singletons which min_observations then
    # discarded, losing people who were plainly visible. Immich's own
    # recognition threshold is 0.5, so this sits deliberately near it.
    distance: float = 0.55
    # "auto": 3 at 2 fps or more, 2 at 1 fps, 1 below that -- at one frame every two
    # seconds a single sighting can be a whole appearance.
    min_observations: int | Literal["auto"] = "auto"


class MatchSettings(BaseModel):
    max_distance: float = 0.4
    top_k: int = 5
    named_people_only: bool = False
    # Minimum lead over the runner-up person. Observed on real data: correct
    # matches lead by 0.27-0.34, while near-misses lead by under 0.02, so this
    # separates them far more cleanly than absolute distance alone.
    min_margin: float = 0.05
    # Also score this many of the sharpest individual observations, not just
    # the cluster centroid, and keep the best result. Averaging a whole
    # appearance dilutes clean frontal frames with profile and blurred ones.
    probe_observations: int = 5
    # When the nearest person is an unnamed cluster and a *named* person is
    # within this distance of it, prefer the name. Immich libraries commonly
    # hold one person as both a name and stray unnamed clusters.
    prefer_named_within: float = 0.06


class WritebackSettings(BaseModel):
    enabled: bool = True
    # Record clusters that matched nobody. Purely local -- no person is created
    # until `ivf promote --write`, which groups them across videos first.
    bank_unmatched: bool = True
    # A group must span this many distinct videos before it is offered as a new
    # person. One stranger in one clip is noise; the same face across several
    # is somebody worth naming.
    promote_min_assets: int = 3
    tag: str = "VideoFaceScan/scanned"


class Config(BaseModel):
    immich: ImmichSettings = Field(default_factory=ImmichSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    db: DbSettings = Field(default_factory=DbSettings)
    select: SelectSettings = Field(default_factory=SelectSettings)
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)
    detect: DetectSettings = Field(default_factory=DetectSettings)
    cluster: ClusterSettings = Field(default_factory=ClusterSettings)
    match: MatchSettings = Field(default_factory=MatchSettings)
    writeback: WritebackSettings = Field(default_factory=WritebackSettings)
    objects: ObjectSettings = Field(default_factory=ObjectSettings)
    timeline: TimelineSettings = Field(default_factory=TimelineSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    sensitive: SensitiveSettings = Field(default_factory=SensitiveSettings)

    state_db: Path = Path("ivf-state.sqlite3")
    report_dir: Path = Path("reports")
    # Set by resolve_auto(): whether "auto" settings were sized for a GPU.
    gpu_resolved: bool | None = None

    def resolve_auto(self) -> "Config":
        """Replace "auto" settings with concrete values, before anything uses them.

        Done once at load, so the fingerprint covers the values actually used:
        moving to or from a GPU changes the sampling and invalidates old scans.
        """
        autos = (
            self.sampling.min_fps == "auto"
            or self.cluster.min_observations == "auto"
            or self.timeline.gap_s == "auto"
        )
        if not autos:
            return self
        if self.sampling.min_fps == "auto":
            self.gpu_resolved = detect_gpu(self.detect)
            self.sampling.min_fps = (
                self.sampling.gpu_fps if self.gpu_resolved else self.sampling.cpu_fps
            )
        fps = float(self.sampling.min_fps)
        if self.cluster.min_observations == "auto":
            self.cluster.min_observations = 3 if fps >= 2 else 2 if fps >= 1 else 1
        if self.timeline.gap_s == "auto":
            self.timeline.gap_s = round(max(3.0, 2.5 / max(fps, 0.01)), 3)
        return self

    @cached_property
    def fingerprint(self) -> str:
        """Hash of everything that changes results.

        Stored per asset so that tuning a threshold correctly invalidates prior
        scans, while re-running with identical settings stays a no-op.
        """
        payload = self._payload()
        # Only when switched on, so turning nothing on leaves existing
        # fingerprints (and therefore finished scans) valid. The search model is
        # left out: it does not change faces or timelines, and a new one is
        # applied to finished videos by `ivf reembed`, frames only, rather than
        # by scanning every video again.
        if self.objects.enabled:
            payload["objects"] = self.objects.model_dump(exclude={"model"})
        return self._hash(payload)

    def _payload(self) -> dict:
        return {
            # `select` belongs here because live_photos changes which *asset* a
            # face is written to, not merely which videos are looked at. Leaving
            # it out let runs staged under the old behaviour pass the staleness
            # check while still pointing at hidden clips.
            "select": self.select.model_dump(),
            # Effective values only: the gpu/cpu presets, the GPU guess and how the
            # ML service is reached do not change results; the resolved rate does.
            "sampling": self.sampling.model_dump(exclude={"gpu_fps", "cpu_fps"}),
            "detect": self.detect.model_dump(
                exclude={"ml_url", "ml_workers", "gpu"}
                | ({"backend"} if self.detect.backend == "insightface" else set())
            ),
            "cluster": self.cluster.model_dump(),
            "match": self.match.model_dump(),
        }

    def legacy_fingerprint(self, model: str) -> str | None:
        """The fingerprint these settings had when it still included the search
        model (`objects.model`), for that model. Lets scans made then count as
        done now (see State.rekey)."""
        if not self.objects.enabled:
            return None
        payload = self._payload()
        payload["objects"] = self.objects.model_copy(update={"model": model}).model_dump()
        return self._hash(payload)

    def _hash(self, payload: dict) -> str:
        if self.index.enabled:
            payload["timeline"] = self.timeline.model_dump()
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def detect_gpu(detect: DetectSettings) -> bool:
    """Best guess at whether face detection will run on a GPU (see DetectSettings.gpu)."""
    if detect.gpu != "auto":
        return bool(detect.gpu)
    if detect.backend == "insightface":
        if detect.device != "cuda":
            return False
        try:
            from .faces import preload_cuda_libraries

            preload_cuda_libraries()
            import onnxruntime

            return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        except Exception:
            return False
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        return False
    try:
        return subprocess.run([nvidia, "-L"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def _apply_env(data: dict) -> dict:
    """Environment overrides for the two values that should not live in a file."""
    if key := os.environ.get("IMMICH_API_KEY"):
        data.setdefault("immich", {})["api_key"] = key
    if pw := os.environ.get("IMMICH_DB_PASSWORD"):
        data.setdefault("db", {})["password"] = pw
    return data


def load_config(path: Path | None = None) -> Config:
    candidates = [path] if path else list(DEFAULT_CONFIG_PATHS)
    for candidate in candidates:
        if candidate and candidate.exists():
            with candidate.open("rb") as handle:
                return Config.model_validate(_apply_env(tomllib.load(handle))).resolve_auto()
    if path:
        raise FileNotFoundError(f"Config file not found: {path}")
    return Config.model_validate(_apply_env({})).resolve_auto()
