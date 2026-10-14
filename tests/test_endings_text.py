"""d2 endings text (spec 2026-10-13 Addendum T2): template, compose prompt d5, rejects, ending rewrite, seeding."""
import re

import pytest

from rover_vlm.descriptions import (
    PROMPT_VERSION, behind_clause, compose_prompt, d2_rejects, description_rejects, ending_rejects,
    template_description,
)

PASS_SOFA = {"type": "pass", "side": "right", "objects": ["f:b1"], "arc_m": 1.0}
WALL_SOFA = {"type": "wall", "side": "left", "objects": ["f:b1"], "arc_m": 1.0}
TURN_L = {"type": "turn", "side": "left", "relation": "behind", "objects": ["f:b1"], "arc_m": 3.0,
          "turn_xz": [0, 3], "pre_heading_xz": [0, 1], "gt_lateral_m": 1.0}
BEHIND_SOFA = {"type": "behind", "relation": "behind", "amount": None, "distance_m": None, "object": "f:b1",
               "disappear_arc_m": 2.4}
BEHIND_TABLE = {**BEHIND_SOFA, "object": "f:b2"}
TURN_END = {"type": "turn", "relation": None, "amount": None, "distance_m": None, "object": None,
            "disappear_arc_m": 2.8}
NEAR_TABLE = {"type": "near", "relation": "in front of", "amount": None, "distance_m": None, "object": "f:b2"}
NAMES = {"f:b1": "grey sofa", "f:b2": "dining table"}


def test_prompt_version():
    assert PROMPT_VERSION == "d5"


@pytest.mark.parametrize("events,stop,text", [
    ([PASS_SOFA], BEHIND_SOFA, "Go right of the grey sofa, then proceed behind it."),
    ([PASS_SOFA], BEHIND_TABLE, "Go right of the grey sofa, then proceed behind the dining table."),
    ([], BEHIND_TABLE, "Go straight ahead, then proceed behind the dining table."),
    ([PASS_SOFA, TURN_L], TURN_END, "Go right of the grey sofa, then turn left behind the grey sofa."),
])
def test_templates_end_with_the_ending_and_pass_their_own_rules(events, stop, text):
    assert template_description(events, stop, NAMES) == text
    assert description_rejects(text, events, stop, NAMES) == []


def test_behind_clause_verb_and_target():
    assert behind_clause([PASS_SOFA], BEHIND_SOFA, NAMES, "carry on") == "carry on behind it"
    assert behind_clause([PASS_SOFA], BEHIND_TABLE, NAMES) == "proceed behind the dining table"


def test_a_behind_ending_never_says_stop_and_says_behind():
    t = "Go right of the grey sofa, then stop well past it."
    assert {"ending:stop_word", "ending:behind", "ending:name"} <= set(description_rejects(t, [PASS_SOFA], BEHIND_SOFA, NAMES))
    assert ending_rejects(t, BEHIND_SOFA) == ["ending:stop_word", "ending:behind"]
    assert ending_rejects(t, NEAR_TABLE) == []                      # visible stops are not ending-checked


def test_it_only_for_an_object_of_the_last_step():
    t = "Go right of the grey sofa, then proceed behind it."
    assert "ending:name" in description_rejects(t, [PASS_SOFA], BEHIND_TABLE, NAMES)


def test_a_turn_ending_forbids_stop_words():
    assert ending_rejects("Go right of the grey sofa, then turn left behind it and stop.", TURN_END) == ["ending:stop_word"]


def test_d2_rejects_combines_turn_and_ending_rules():
    ev = {"events": [PASS_SOFA], "stop": BEHIND_SOFA}
    assert d2_rejects("Go right of the grey sofa, then turn left behind it.", ev) == ["turn:stray"]
    assert d2_rejects("Go right of the grey sofa, then stop.", ev) == ["ending:stop_word", "ending:behind"]


def _norm(p):
    return " ".join(p.split())


def test_compose_prompt_d5_endings():
    p = _norm(compose_prompt([PASS_SOFA], BEHIND_TABLE, NAMES, "plain", "one sentence"))
    assert "Then: proceed behind the dining table." in p and "never say where it stops" in p
    assert "Keep where to stop." not in p
    t = _norm(compose_prompt([PASS_SOFA, TURN_L], TURN_END, NAMES, "plain", "one sentence"))
    assert "Then:" not in t and "End with the turn; never say where it stops." in t
    s = _norm(compose_prompt([PASS_SOFA], NEAR_TABLE, NAMES, "plain", "one sentence"))
    assert "Then: stop in front of the dining table." in s and "Keep where to stop." in s
