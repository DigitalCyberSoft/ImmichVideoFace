"""A rescan keeps search frames that a scan with the current settings would redo identically."""

from types import SimpleNamespace

from immich_video_face.config import Config
from immich_video_face.scanner import Scanner
from immich_video_face.video_index import frames_reusable


def scanner(fps: float, interval_s: float = 2.0, clip: bool = True) -> Scanner:
    cfg = Config.model_validate(
        {"sampling": {"min_fps": fps}, "objects": {"enabled": True, "interval_s": interval_s}}
    )
    detector = SimpleNamespace(clip_model="ViT-B-32__openai" if clip else None)
    return Scanner(cfg, detector, matcher=None)


def test_face_rate_change_keeps_the_search_spacing():
    assert scanner(0.5).clip_spacing_s() == 2.0
    assert scanner(1.0).clip_spacing_s() == 2.0
    assert scanner(8.0).clip_spacing_s() == 2.0
    assert scanner(1.0).clip_every() == 2


def test_no_search_model_means_no_search_frames():
    assert scanner(1.0, clip=False).clip_every() == 0


def reusable(**kw) -> bool:
    args = dict(count=30, same_model=True, median_gap_ms=2002, duration_ms=60_000, spacing_ms=2000)
    args.update(kw)
    return frames_reusable(**args)


def test_same_model_and_spacing_are_kept():
    assert reusable()


def test_other_model_is_redone():
    assert not reusable(same_model=False)


def test_other_spacing_is_redone():
    assert not reusable(median_gap_ms=4000)
    assert not reusable(median_gap_ms=1000)


def test_nothing_indexed_is_redone():
    assert not reusable(count=0, median_gap_ms=None)


def test_single_frame_kept_only_for_a_short_video():
    assert reusable(count=1, median_gap_ms=None, duration_ms=1_800)
    assert not reusable(count=1, median_gap_ms=None, duration_ms=9_000)
    assert not reusable(count=1, median_gap_ms=None, duration_ms=None)
