"""Appearance intervals and face resolution for the video timeline."""

import math

from immich_video_face.cluster import merge_segments, time_segments
from immich_video_face.video_index import _vector_literal, resolve_face_ids


def test_continuous_sightings_make_one_appearance():
    times = [10.0, 10.5, 11.0, 11.5, 12.0]
    assert time_segments(times, gap_s=3.0, pad_s=0.25) == [(9.75, 12.25)]


def test_a_long_absence_starts_a_new_appearance():
    times = [1.0, 1.5, 2.0, 9.0, 9.5]
    assert time_segments(times, gap_s=3.0, pad_s=0.25) == [(0.75, 2.25), (8.75, 9.75)]


def test_short_misses_are_bridged():
    # Three missed frames at 2 fps (a turned head) stay one appearance.
    times = [1.0, 1.5, 3.5, 4.0]
    assert time_segments(times, gap_s=3.0, pad_s=0.25) == [(0.75, 4.25)]


def test_padding_is_clamped_to_the_video():
    assert time_segments([0.0, 0.5], gap_s=3.0, pad_s=0.25, duration_s=0.6) == [(0.0, 0.6)]


def test_unsorted_input_and_missing_timestamps():
    times = [5.0, float("nan"), 4.5, 4.0]
    assert time_segments(times, gap_s=3.0, pad_s=0.0) == [(4.0, 5.0)]
    assert time_segments([math.nan], gap_s=3.0, pad_s=0.0) == []


def test_merge_unions_and_bridges():
    segments = [(5.0, 6.0), (0.0, 2.0), (1.5, 3.0)]
    assert merge_segments(segments) == [(0.0, 3.0), (5.0, 6.0)]
    assert merge_segments(segments, gap_s=2.0) == [(0.0, 6.0)]


def test_resolve_face_ids_prefers_immich_detections():
    faces = [
        {"id": "manual-1", "sourceType": "manual", "person": {"id": "alice"}},
        {"id": "ml-1", "sourceType": "machine-learning", "person": {"id": "alice"}},
        {"id": "manual-2", "sourceType": "manual", "person": {"id": "bob"}},
        {"id": "unassigned", "sourceType": "machine-learning", "person": None},
    ]
    assert resolve_face_ids(faces) == {"alice": "ml-1", "bob": "manual-2"}


def test_vector_literal():
    assert _vector_literal([0.5, -1.0, 1e-9]) == "[0.5,-1,1e-09]"


def test_failed_assets_count_as_done_unless_retrying(tmp_path):
    from immich_video_face.state import State

    with State(tmp_path / "state.sqlite3") as state:
        run = state.start_run("fp", "scan")
        state.record_scan(asset_id="ok", fingerprint="fp", run_id=run, frames=1, observations=0,
                          clusters=0, matches=0, status="ok", error=None)
        state.record_scan(asset_id="bad", fingerprint="fp", run_id=run, frames=0, observations=0,
                          clusters=0, matches=0, status="error", error="moov atom not found")
        state.settle_run(run)
        assert state.already_scanned("fp") == {"ok", "bad"}
        assert state.already_scanned("fp", include_failed=False) == {"ok"}
        assert state.already_scanned("other") == set()
        assert state.failed_assets() == [("bad", "moov atom not found")]
