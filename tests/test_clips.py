"""Run-grouping and criteria for benchmark clip selection (`rover_vlm.clips`).

A clip must be one manoeuvre: contiguous, merged across a momentary straightening, padded
for context, and long enough to be worth watching. These pin that behaviour on synthetic
index entries, plus the eligibility gates that keep the three criteria disjoint in intent.
"""

from pathlib import Path

import pytest

from rover_vlm.clips import CRITERIA_BY_NAME, Criterion, bend, find_runs, goal_overshoot


def _entry(v, path=None, goal=(0.5, 0.5, 1), **meta):
    return {"v": v, "i": v, "t": 100.0 + v * 0.1, "reason": None, "id": f"f{v}",
            "image": f"/tmp/{v}.png",
            "answer": {"path": path or [[0.5, 1.0, 1], [0.5, 0.5, 1]], "goal": list(goal)},
            "meta": {"initial_angle_deg": 0.0, "goal_angle_deg": 0.0, **meta}}


def _curved(v, offset):
    """A path whose waypoints sit `offset` to the right of where they started."""
    return _entry(v, path=[[0.5, 1.0, 1], [0.5 + offset, 0.7, 1], [0.5 + offset, 0.4, 1]])


ALWAYS = Criterion("t", "", score=lambda e: e["meta"]["s"], eligible=lambda e: True, threshold=1.0)


def _scored(v, s):
    e = _entry(v)
    e["meta"]["s"] = s
    return e


def test_bend_is_signed_and_zero_for_a_straight_path():
    assert bend(_entry(0)) == pytest.approx(0.0, abs=1e-9)
    assert bend(_curved(0, 0.2)) > 0.05        # drifts right
    assert bend(_curved(0, -0.2)) < -0.05      # drifts left
    assert bend(_curved(0, 0.2)) == pytest.approx(-bend(_curved(0, -0.2)))


def test_runs_are_contiguous_and_ranked_by_peak():
    entries = [_scored(v, 5.0 if v in (10, 11, 12) else 2.0 if v in (30, 31) else 0.0)
               for v in range(40)]
    runs = find_runs(entries, ALWAYS)
    assert [(r.start, r.end) for r in runs] == [(10, 12), (30, 31)]  # strongest first
    assert runs[0].peak == 5.0 and runs[0].peak_index == 10 and runs[0].n_hits == 3


def test_short_dropout_is_merged_but_a_long_one_splits():
    entries = [_scored(v, 5.0 if v in (10, 11, 14, 15) else 0.0) for v in range(30)]
    assert [(r.start, r.end) for r in find_runs(entries, ALWAYS, merge_gap=3)] == [(10, 15)]
    assert [(r.start, r.end) for r in find_runs(entries, ALWAYS, merge_gap=1)] == [(10, 11), (14, 15)]


def test_padding_adds_context_and_is_clamped_to_the_sequence():
    entries = [_scored(v, 5.0 if v in (1, 2) else 0.0) for v in range(10)]
    r = find_runs(entries, ALWAYS, pad=4)[0]
    assert (r.start, r.end) == (0, 6)              # clamped at 0, not -3
    assert [e["v"] for e in r.entries] == list(range(0, 7))
    assert find_runs(entries, ALWAYS, pad=0)[0].entries[0]["v"] == 1


def test_min_frames_drops_a_blip():
    entries = [_scored(v, 5.0 if v in (4,) else 0.0) for v in range(10)]
    assert find_runs(entries, ALWAYS, min_frames=2) == []
    assert len(find_runs(entries, ALWAYS, min_frames=1)) == 1


def test_curve_criterion_ignores_spins_and_offscreen_goals():
    c = CRITERIA_BY_NAME["curve"]
    assert c.eligible(_curved(0, 0.2)) and c.score(_curved(0, 0.2)) > c.threshold
    assert not c.eligible(_entry(0, goal_offscreen=True))       # that is out_of_view's job
    assert not c.eligible(_entry(0, initial_angle_deg=70.0))    # spinning in place
    assert c.score(_curved(0, -0.2)) == pytest.approx(c.score(_curved(0, 0.2)))  # both ways


def test_occluded_criterion_needs_a_hidden_goal_and_forward_motion():
    c = CRITERIA_BY_NAME["occluded"]
    assert c.eligible(_entry(0, goal=(0.5, 0.5, 0)))
    assert not c.eligible(_entry(0, goal=(0.5, 0.5, 1)))
    # --allow-offscreen also relaxes the initial-heading filter, so spin-in-place frames
    # reach the index; a rover pirouetting is not a path benchmark whatever it occludes
    assert not c.eligible(_entry(0, goal=(0.5, 0.5, 0), initial_angle_deg=75.0))
    # GND labels everything visible, so it can never qualify -- that is the point
    assert find_runs([_entry(v) for v in range(20)], c) == []


def test_out_of_view_prefers_a_goal_just_past_the_edge():
    """Ranking by bearing picks goals 85 deg off axis with no route left in view; the
    useful case is a goal barely outside the frame."""
    c = CRITERIA_BY_NAME["out_of_view"]
    near = _entry(0, goal_offscreen=True, goal_uv_norm=[1.05, 0.5], goal_angle_deg=-50.0)
    far = _entry(1, goal_offscreen=True, goal_uv_norm=[9.0, 0.5], goal_angle_deg=-88.0)
    assert goal_overshoot(near) == pytest.approx(0.05)
    assert c.eligible(near) and c.score(near) > c.threshold
    assert not c.eligible(far), "a goal eight frame-widths out is not 'just off screen'"
    assert c.score(near) > c.score(_entry(0, goal_offscreen=True, goal_uv_norm=[1.5, 0.5]))
    assert not c.eligible(_entry(0, goal_uv_norm=[1.05, 0.5]))          # on screen
    assert not c.eligible(_entry(0, goal_offscreen=True, goal_uv_norm=[1.05, 0.5],
                                 initial_angle_deg=80.0))               # spinning


# --- eval-set paths: the bug that killed job 90719 --------------------------------------

def _find_clips_mod():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "find_clips", Path(__file__).resolve().parent.parent / "scripts" / "find_clips.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["find_clips"] = m
    spec.loader.exec_module(m)
    return m


def test_resolve_image_anchors_relative_paths_to_the_repo_root():
    """scripts/evaluate.py opens `IMAGE_ROOT / rec["image"][0]`, so a relative path in an
    eval record silently becomes a path under data/sharerobot/trajectory and the job dies
    on its first sample. Records must carry absolute paths."""
    fc = _find_clips_mod()
    root = fc.REPO_ROOT
    rel = fc.resolve_image("data/real/tum/seq/rgb/1.png")
    assert rel == str(root / "data/real/tum/seq/rgb/1.png")
    assert Path(rel).is_absolute()
    # an already-absolute entry (GND frames) must survive untouched
    abs_in = str(root / "outputs/videos/_frames/bag/1.jpg")
    assert fc.resolve_image(abs_in) == abs_in
    # and the result must be a no-op under evaluate.py's join, whatever IMAGE_ROOT is
    for got in (rel, abs_in):
        assert str(Path("/some/image/root") / got) == got
