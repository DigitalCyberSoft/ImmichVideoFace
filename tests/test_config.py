"""The "auto" sampling settings follow whether detection runs on a GPU."""

from immich_video_face.config import Config


def resolved(gpu: bool, **overrides) -> Config:
    data = {"detect": {"backend": "immich-ml", "gpu": gpu}}
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return Config.model_validate(data).resolve_auto()


def test_cpu_defaults_to_one_frame_every_two_seconds():
    cfg = resolved(gpu=False)
    assert cfg.sampling.min_fps == 0.5
    assert cfg.cluster.min_observations == 1
    assert cfg.timeline.gap_s == 5.0
    assert cfg.gpu_resolved is False


def test_gpu_defaults_to_eight_fps():
    cfg = resolved(gpu=True)
    assert cfg.sampling.min_fps == 8.0
    assert cfg.cluster.min_observations == 3
    assert cfg.timeline.gap_s == 3.0


def test_explicit_values_win():
    cfg = resolved(gpu=False, sampling={"min_fps": 2.0}, cluster={"min_observations": 4})
    assert cfg.sampling.min_fps == 2.0
    assert cfg.cluster.min_observations == 4
    assert cfg.timeline.gap_s == 3.0


def test_fingerprint_tracks_the_resolved_rate():
    assert resolved(gpu=False).fingerprint != resolved(gpu=True).fingerprint
    assert resolved(gpu=False).fingerprint == resolved(gpu=False).fingerprint


def test_fingerprint_ignores_how_the_ml_service_is_reached():
    a = resolved(gpu=False, detect={"ml_workers": 2})
    b = resolved(gpu=False, detect={"ml_workers": 8, "ml_url": "http://ml:3003"})
    assert a.fingerprint == b.fingerprint
