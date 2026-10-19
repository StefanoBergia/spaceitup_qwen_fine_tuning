"""Naming chooser (spec 2026-10-15 Addendum C)."""
import json

import pytest

from rover_vlm.descriptions import (candidate_names, choose_object, choose_prompt, parse_choice_reply,
                                    summarize_d1)


def _norm(p):
    return " ".join(p.split())


def _r(choice):
    return f"<answer>{json.dumps({'choice': choice})}</answer>"


def test_candidate_names_distinct_and_stable():
    vn = {"V0": {"o": "Grey Sofa"}, "V1": {"o": "grey sofa"}, "V2": {"o": "white wall"}, "V3": {"o": None}}
    got = candidate_names("o", vn)
    assert sorted(c["name"] for c in got) == ["Grey Sofa", "white wall"]
    assert next(c for c in got if c["name"] == "Grey Sofa")["from"] == ["V0", "V1"]
    assert got == candidate_names("o", vn)
    assert candidate_names("missing", vn) == []


def test_choose_prompt_c1():
    p = _norm(choose_prompt("wall", ["white wall", "kitchen island"]))
    assert "a. white wall" in p and "b. kitchen island" in p
    assert "Pick the one name that best names the single thing that fills most of the outline" in p
    assert "Look at what is inside the outline, not at what is next to it or behind it." in p
    assert 'If no name fits, answer "none".' in p
    assert "something long that the rover drives alongside or stops in front of" in p
    assert '{"choice": "<letter>|none"}' in p
    with pytest.raises(ValueError):
        choose_prompt("wall", ["x"], "c9")


def test_parse_choice_reply():                                       # Review Focus 1
    assert parse_choice_reply(_r("b"), 3) == "b"
    assert parse_choice_reply(_r("B"), 3) == "b"
    assert parse_choice_reply(_r("c. white wall"), 3) == "c"
    assert parse_choice_reply("<answer>b</answer>", 3) == "b"              # a bare answer counts
    assert parse_choice_reply(_r("none"), 3) == "none"
    assert parse_choice_reply(_r("None of these"), 3) == "none"
    for bad in ("d", "b, c", "white wall", ""):                            # past the list, two, a name, empty
        assert parse_choice_reply(_r(bad), 3) is None, bad
    assert parse_choice_reply("I think it is a wall", 3) is None


OBJ = {"object_id": "o", "kind_hint": "wall", "image": None, "polygon": [], "crop_box": None}
CANDS = [{"name": "white wall", "from": ["V2", "V3"]}, {"name": "grey sofa", "from": ["V0"]}]


def _ask(*replies):
    calls = []

    def ask(img, prompt, seed):
        calls.append(seed)
        return replies[min(len(calls), len(replies)) - 1], "stop"
    return ask, calls


def test_choose_object_pick_none_and_failure():
    ask, calls = _ask(_r("b"))
    row = choose_object(ask, OBJ, CANDS, "kitchen island", seed=0, image=b"x")
    assert (row["name"], row["name_source"], row["passed"], row["attempts"]) == ("grey sofa", ["V0"], True, 1)
    assert row["candidates"] == ["white wall", "grey sofa"] and row["need"] == "wall" and row["choose_prompt"] == "c1"
    ask, _ = _ask(_r("none"))
    row = choose_object(ask, OBJ, CANDS, "kitchen island", seed=0, image=b"x")
    assert (row["choice"], row["name"], row["name_source"], row["passed"]) == ("none", "kitchen island", ["V0_fallback"], True)
    ask, calls = _ask("garbage", "more garbage")
    row = choose_object(ask, OBJ, CANDS, "kitchen island", seed=0, image=b"x")
    assert (row["choice"], row["rejects"], row["attempts"], row["name"]) == (None, ["parse"], 2, "kitchen island")
    assert calls == [0, 1] and "raw" in row
    ask, _ = _ask("garbage")
    assert choose_object(ask, OBJ, CANDS, None, seed=0, image=b"x")["passed"] is False      # Review Focus 2


def test_choose_object_single_and_no_candidates_make_no_call():
    ask, calls = _ask(_r("a"))
    row = choose_object(ask, OBJ, CANDS[:1], "kitchen island", seed=0, image=b"x")
    assert (row["single"], row["choice"], row["attempts"], row["name"]) == (True, "a", 0, "white wall")
    row = choose_object(ask, OBJ, [], "kitchen island", seed=0, image=b"x")
    assert (row["name"], row["name_source"], row["passed"]) == ("kitchen island", ["V0_fallback"], True)
    assert calls == []


def test_summarize_choose():
    base = {"passed": True, "rejects": [], "single": False, "candidates": ["x", "y"]}
    rows = [{**base, "id": "1", "choice": "a", "name_source": ["V2"]},
            {**base, "id": "2", "choice": "none", "name_source": ["V0_fallback"]},
            {**base, "id": "3", "choice": None, "rejects": ["parse"], "name_source": ["V0_fallback"]},
            {**base, "id": "4", "choice": "a", "single": True, "candidates": ["x"], "name_source": ["V1", "V2"]},
            {**base, "id": "5", "choice": None, "candidates": [], "name_source": ["V0_fallback"]}]
    s = summarize_d1(rows, "choose")
    assert s["choices"] == {"picked": 1, "none": 1, "failed": 1, "single": 1, "no_candidates": 1}
    assert s["from"] == {"V2": 2, "V0_fallback": 3, "V1": 1}
