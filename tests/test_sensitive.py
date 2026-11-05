"""Sensitive-content settings: categories are configurable without restating prompts."""

import pytest
from pydantic import ValidationError

from immich_video_face.config import Config


def test_defaults_flag_explicit_only():
    cfg = Config.model_validate({"sensitive": {"enabled": True}})
    assert cfg.sensitive.flag == ["Explicit"]
    assert {"Explicit", "Nudity", "Underwear"} <= set(cfg.sensitive.categories)


def test_overriding_a_cut_keeps_the_prompts():
    cfg = Config.model_validate(
        {"sensitive": {"categories": {"Explicit": {"min_margin": 0.03}}}}
    )
    explicit = cfg.sensitive.categories["Explicit"]
    assert explicit.min_margin == 0.03
    assert explicit.prompts and explicit.contrast
    assert "Nudity" in cfg.sensitive.categories


def test_a_new_category_needs_its_prompts():
    cfg = Config.model_validate(
        {
            "sensitive": {
                "flag": ["Explicit", "Gore"],
                "categories": {"Gore": {"prompts": ["a photo of gore"], "contrast": ["a photo of food"]}},
            }
        }
    )
    assert cfg.sensitive.categories["Gore"].prompts == ["a photo of gore"]
    with pytest.raises(ValidationError):
        Config.model_validate({"sensitive": {"categories": {"Gore": {"min_margin": 0.1}}}})


def test_flagging_an_unknown_category_is_refused():
    with pytest.raises(ValidationError):
        Config.model_validate({"sensitive": {"flag": ["Explict"]}})


def test_sensitive_settings_do_not_touch_the_scan_fingerprint():
    plain = Config.model_validate({})
    flagged = Config.model_validate({"sensitive": {"enabled": True, "flag": ["Nudity"]}})
    assert plain.fingerprint == flagged.fingerprint


def plan(monkeypatch, tmp_path, *, scores, current, ours=(), cleared=(), flag=("Explicit",)):
    from immich_video_face import sensitive
    from immich_video_face.state import State

    state = State(tmp_path / "state.sqlite3")
    state.sensitive_record("Explicit", list(ours) + list(cleared), "tag-1")
    for asset_id in cleared:
        state.sensitive_mark_cleared("Explicit", asset_id)
    monkeypatch.setattr(sensitive, "tagged_with", lambda conn, owner, name: set(current))
    monkeypatch.setattr(sensitive, "score_assets", lambda conn, cfg, owner, category: dict(scores))
    cfg = Config.model_validate({"sensitive": {"enabled": True, "flag": list(flag)}})
    return sensitive.plan_category(None, state, cfg, "owner", "Explicit", "Explicit" in flag)


def test_plan_tags_what_passes(monkeypatch, tmp_path):
    p = plan(monkeypatch, tmp_path, scores={"a": 0.05, "b": -0.2}, current=[])
    assert p.to_tag == ["a"] and p.to_untag == []
    assert p.tag_name == "Sensitive/Explicit"


def test_a_tag_removed_by_hand_is_not_put_back(monkeypatch, tmp_path):
    p = plan(monkeypatch, tmp_path, scores={"a": 0.05}, current=[], ours=["a"])
    assert p.cleared_by_hand == ["a"] and p.to_tag == []


def test_a_tag_added_by_hand_is_left_alone(monkeypatch, tmp_path):
    p = plan(monkeypatch, tmp_path, scores={"a": -0.5}, current=["a"])
    assert p.to_untag == [] and p.to_tag == []


def test_our_tag_comes_off_when_the_asset_stops_passing(monkeypatch, tmp_path):
    p = plan(monkeypatch, tmp_path, scores={"a": -0.5}, current=["a"], ours=["a"])
    assert p.to_untag == ["a"]


def test_unflagging_a_category_removes_only_our_tags(monkeypatch, tmp_path):
    p = plan(monkeypatch, tmp_path, scores={}, current=["a", "manual"], ours=["a"], flag=())
    assert p.to_untag == ["a"]
