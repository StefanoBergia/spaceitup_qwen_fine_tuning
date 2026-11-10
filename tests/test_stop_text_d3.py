"""d3 stop text (spec 2026-11-10-stop-phrases-d3-design.md): object relations and places, distance words rejected,
the stop kind in desc_check, prompt d6."""

import pytest

from rover_vlm.descriptions import (
    PLACE_TEXT, PLACE_WORDS, PROMPT_VERSION, _check_ending, compose_prompt, d2_rejects, d3_rejects,
    description_rejects, stop_text, template_description, visible_stop_rejects,
)

PASS_SOFA = {"type": "pass", "side": "left", "objects": ["f:b1"], "arc_m": 1.0}
NAMES = {"f:b1": "sofa", "f:b2": "table"}
BEHIND_SOFA = {"type": "behind", "relation": "behind", "amount": None, "distance_m": None, "object": "f:b1",
               "disappear_arc_m": 2.4}


def _obj(rel):
    return {"type": "object", "relation": rel, "amount": None, "distance_m": None, "object": "f:b2", "place": None}


def _place(p):
    return {"type": "place", "relation": None, "amount": None, "distance_m": None, "object": None, "place": p,
            "measures": {}}


D3_STOPS = [_obj(r) for r in ("behind", "in front of", "next to")] + [_place(p) for p in PLACE_TEXT] + \
    [{**_obj("in front of"), "type": "wall"}]


@pytest.mark.parametrize("stop,text", [
    (_obj("next to"), "stop next to the table"),
    (_obj("behind"), "stop behind the table"),
    (_place("room_middle"), "stop in the middle of the room"),
    (_place("hallway_end"), "stop at the end of the hallway"),
    (_place("corner"), "stop in the corner"),
    (_place("wall"), "stop by the wall"),
    (_place("hallway"), "stop in the hallway"),
    ({"type": "wall", "relation": "in front of", "object": "f:b2"}, "stop in front of the table"),
    (_place("doorway"), "stop in the doorway")])
def test_stop_text_d3(stop, text):
    assert stop_text(stop, "table") == text


def test_place_words_cover_every_place():
    assert set(PLACE_TEXT) == set(PLACE_WORDS) == {"doorway", "corner", "hallway_end", "hallway", "wall", "room_middle"}


@pytest.mark.parametrize("stop", D3_STOPS, ids=lambda s: s["relation"] or s["place"])
def test_template_passes_rejects_for_every_d3_stop(stop):
    text = template_description([PASS_SOFA], stop, NAMES)
    assert description_rejects(text, [PASS_SOFA], stop, NAMES) == []


def test_distance_words_rejected():
    r = description_rejects("Go left of the sofa, then stop well past the sofa.", [PASS_SOFA], _place("room_middle"), NAMES)
    assert "stop:distance" in r and "stop:place" in r


def test_distance_words_rejected_for_object_stops():
    r = description_rejects("Go left of the sofa, then stop a little before the table, in front of it.",
                            [PASS_SOFA], _obj("in front of"), NAMES)
    assert "stop:distance" in r


def test_place_word_required():
    r = description_rejects("Go left of the sofa, then stop by the window.", [PASS_SOFA], _place("corner"), NAMES)
    assert "stop:place" in r


def test_variant_place_words_pass():
    t = "Go left of the sofa, then stop at the end of the corridor."
    assert description_rejects(t, [PASS_SOFA], _place("hallway_end"), NAMES) == []


def test_next_to_accepts_beside():
    t = "Go left of the sofa, then stop beside the table."
    assert description_rejects(t, [PASS_SOFA], _obj("next to"), NAMES) == []


def test_visible_stop_rejects_is_names_free():
    assert visible_stop_rejects("Go left of the sofa, then stop in the middle of the room.", _place("room_middle")) == []
    assert visible_stop_rejects("Go left of the sofa, then stop well past it.", _place("room_middle")) == \
        ["stop:distance", "stop:place"]
    assert visible_stop_rejects("Go left of the sofa.", _place("room_middle")) == ["stop:missing"]


def test_d3_rejects_leaves_behind_and_turn_texts_alone():          # Review Focus 3
    ev = {"events": [PASS_SOFA], "stop": BEHIND_SOFA}
    t = "Go left of the sofa, then carry on behind it."
    assert d3_rejects(t, ev) == d2_rejects(t, ev) == []


def test_d3_rejects_flags_a_reused_distance_stop():
    ev = {"events": [PASS_SOFA], "stop": _place("corner")}
    assert "stop:distance" in d3_rejects("Go left of the sofa, then stop well past the sofa.", ev)


def test_check_ending_carries_the_stop_kind():
    assert _check_ending(_place("corner"), {}) == {"type": "stop", "stop": "place:corner"}
    assert _check_ending(_obj("next to"), {}) == {"type": "stop", "stop": "object:next to"}
    near = {"type": "near", "relation": "in front of", "amount": None, "distance_m": None, "object": "f:b2"}
    assert _check_ending(near, {}) == {"type": "stop"}


def test_compose_prompt_has_the_new_stop_line():
    p = " ".join(compose_prompt([PASS_SOFA], _obj("next to"), NAMES, "plain", "one sentence").split())
    assert "Then: stop next to the table." in p and "Keep where to stop." in p
    q = " ".join(compose_prompt([PASS_SOFA], _place("room_middle"), NAMES, "plain", "one sentence").split())
    assert "Then: stop in the middle of the room." in q


def test_prompt_version():
    assert PROMPT_VERSION == "d6"



def test_hallway_words_are_not_the_end_of_the_hallway():
    t = "Go left of the sofa, then stop at the end of the hallway."
    assert "stop:place" in description_rejects(t, [PASS_SOFA], _place("hallway"), NAMES)


def test_wall_stop_needs_in_front_of():
    wall = {**_obj("in front of"), "type": "wall"}
    assert visible_stop_rejects("Go left of the sofa, then stop beside the table.", wall) == ["stop:relation"]
    assert _check_ending(wall, {}) == {"type": "stop", "stop": "wall:in front of"}
