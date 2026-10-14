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


from rover_vlm.descriptions import REWRITE_VERBS, rewrite_ending, seed_d2_labels   # noqa: E402

VERB = "(proceed|continue|carry on|go on)"
REF = {"f:b1": "beige sofa", "f:b3": "refrigerator", "f:b4": "living room", "f:b5": "fireplace",
       "f:b6": "frosted glass panel"}


def _pass(oid, side="right"):
    return {"type": "pass", "side": side, "objects": [oid], "arc_m": 1.0}


def _behind(oid):
    return {**BEHIND_SOFA, "object": oid}


@pytest.mark.parametrize("text,events,oid,want", [
    ("Keep the beige sofa on your left, then stop a little past the end of the beige sofa.",
     [{"type": "wall", "side": "left", "objects": ["f:b1"], "arc_m": 1.0}], "f:b1",
     rf"Keep the beige sofa on your left, then {VERB} behind it\."),
    ("Go right of the refrigerator and stop well past the refrigerator.", [_pass("f:b3")], "f:b3",
     rf"Go right of the refrigerator, then {VERB} behind it\."),
    ("Go right of the living room. Then stop a little past the living room.", [_pass("f:b4")], "f:b4",
     rf"Go right of the living room, then {VERB} behind it\."),
    ("Could you please go left of the fireplace, then stop well past it?", [_pass("f:b5", "left")], "f:b5",
     rf"Could you please go left of the fireplace, then {VERB} behind it\?"),                     # Review Focus 5
    ("Please go left of the frosted glass panel, then stop well past the frosted glass panel, as requested.",
     [_pass("f:b6", "left")], "f:b6", rf"Please go left of the frosted glass panel, then {VERB} behind it\."),
])
def test_rewrite_ending(text, events, oid, want):
    new, why = rewrite_ending(text, events, _behind(oid), REF, "route-1")
    assert why is None and re.fullmatch(want, new), new
    assert rewrite_ending(text, events, _behind(oid), REF, "route-1") == (new, None)       # stable per route


def test_rewrite_verbs_come_from_the_route_id():
    assert REWRITE_VERBS == ("proceed", "continue", "carry on", "go on")
    t = "Go right of the refrigerator, then stop well past it."
    got = {rewrite_ending(t, [_pass("f:b3")], _behind("f:b3"), REF, f"r{i}")[0].split(", then ")[1] for i in range(60)}
    assert got == {f"{v} behind it." for v in REWRITE_VERBS}


@pytest.mark.parametrize("text,why", [
    ("Stop well past the refrigerator after going right of it.", "no_connector"),            # Review Focus 2
    ("Go right of the refrigerator.", "no_stop_clause"),
    ("Go right of the fridge, then stop well past it.", "rejects"),                          # the d1 name differs
])
def test_rewrite_ending_refuses(text, why):
    assert rewrite_ending(text, [_pass("f:b3")], _behind("f:b3"), REF, "r") == (None, why)


STOP_NEAR = {"type": "near", "relation": "in front of", "amount": None, "distance_m": None, "object": "f:b2"}


def _ev(rid, events, stop=STOP_NEAR, reason=None):
    return {"id": rid, "reason": reason, "events": events, "stop": stop, "pair_id": None, "kind": "instructed"}


def test_seed_rewrites_unchanged_routes_that_now_end_behind():
    p = _pass("f:b1", "left")
    old = [_ev("a", [p]), _ev("b", [p]), _ev("c", [p]), _ev("d", [p])]
    new = [_ev("a", [p], _behind("f:b1")),             # rewrite
           _ev("b", [], _behind("f:b1")),              # its pass was dropped: events changed -> compose
           _ev("c", [p], _behind("f:b9")),             # X has no d1 name -> compose
           _ev("d", [p])]                              # unchanged -> reuse
    objects = [{"object_id": o} for o in ("f:b1", "f:b2")]
    names = {"gemma": [{"id": "f:b1", "name": "sofa", "passed": True}, {"id": "f:b2", "name": "fridge", "passed": True}],
             "cosmos": []}
    desc = [{"id": i, "description": "Go left of the sofa, then stop in front of the fridge.", "passed": True,
             "namer": "primary", "instruction_source": "cosmos"} for i in "abcd"]
    s = seed_d2_labels(old, new, objects, names, desc)
    by = {r["id"]: r for r in s["desc"]}
    assert set(by) == {"a", "d"} and by["d"]["reused_from"] == "d1" and by["a"]["rewritten_from"] == "d1"
    assert re.fullmatch(rf"Go left of the sofa, then {VERB} behind it\.", by["a"]["description"])
    assert by["a"]["passed"] is True and by["a"]["instruction_source"] == "cosmos"
    assert [r["id"] for r in s["changed"]] == ["b", "c"]
    assert s["rewrite_failed"] == {"events_changed": 1, "unnamed": 1}
