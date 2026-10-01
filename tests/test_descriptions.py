"""Prompt d1: templates, text rules, name rules, Cosmos reply handling, finalize."""
import pytest

from rover_vlm.descriptions import (
    description_rejects, name_rejects, template_description,
)

NAMES = {"f:b1": "coffee table", "f:b2": "stone column", "f:b3": "grey sofa",
         "f:gap4": "open doorway", "f:b5": "fridge", "f:b6": "table"}
PASS_L = {"type": "pass", "side": "left", "objects": ["f:b1"]}
PASS_R = {"type": "pass", "side": "right", "objects": ["f:b3"]}
BETWEEN = {"type": "between", "side": None, "objects": ["f:b2", "f:b3"]}
THROUGH = {"type": "through", "side": None, "objects": ["f:gap4"]}
NEAR = {"type": "near", "relation": "in front of", "distance_m": None, "object": "f:b5"}
AFTER = {"type": "after_last", "relation": "just after", "distance_m": None, "object": None}
DIST = {"type": "distance", "relation": "before", "distance_m": 2.0, "object": "f:b5"}


@pytest.mark.parametrize("events,stop", [
    ([PASS_L], NEAR), ([PASS_L, THROUGH], NEAR), ([BETWEEN], DIST), ([PASS_L, PASS_R], AFTER), ([], NEAR),
])
def test_template_always_passes_its_own_rules(events, stop):
    text = template_description(events, stop, NAMES)
    assert description_rejects(text, events, stop, NAMES) == [], text


def test_template_wording():
    assert template_description([PASS_L, THROUGH], NEAR, NAMES) == (
        "Go left of the coffee table, then go through the open doorway, and stop in front of the fridge.")
    assert template_description([BETWEEN], DIST, NAMES).endswith("stop about two metres before the fridge.")


def test_paraphrases_with_both_side_word_positions_pass():
    ev, st = [PASS_L, PASS_R], NEAR
    ok = ["Keep left of the coffee table, then pass the grey sofa on the right and stop in front of the fridge.",
          "Pass the coffee table on the left, then the grey sofa on the right, and stop in front of the fridge."]
    for t in ok:
        assert description_rejects(t, ev, st, NAMES) == [], t


@pytest.mark.parametrize("text,reason", [
    ("Go right of the coffee table and stop in front of the fridge.", "side:missing:0"),
    ("Go left of the coffee table and stop by the fridge on the left.", "side:stray"),
    ("Stop in front of the fridge.", "name:missing:0"),
    ("Go left of the coffee table and stop.", "stop:name"),
    ("Go left of the coffee table, then the fridge.", "stop:missing"),
    ("Go left of the coffee table on your left and stop in front of the fridge.", "side:egocentric"),
    ("Follow the magenta line left of the coffee table and stop at the fridge.", "drawing:magenta"),
])
def test_rejections(text, reason):
    assert reason in description_rejects(text, [PASS_L], NEAR, NAMES)


def test_names_must_come_in_route_order():
    t = "Stop in front of the fridge after going left of the coffee table."
    assert "stop:name" in description_rejects(t, [PASS_L], NEAR, NAMES)


def test_same_head_noun_twice_matches_in_order():
    names = dict(NAMES, **{"f:b1": "coffee table", "f:b3": "dining table"})
    ev = [PASS_L, {"type": "pass", "side": "right", "objects": ["f:b3"]}]
    t = "Go left of the coffee table, then right of the dining table, and stop in front of the fridge."
    assert description_rejects(t, ev, NEAR, names) == []
    t2 = "Go left of the coffee table and stop in front of the fridge."
    assert "name:missing:1" in description_rejects(t2, ev, NEAR, names)


def test_distance_and_after_rules():
    assert "stop:distance" in description_rejects(
        "Go between the stone column and the grey sofa and stop before the fridge.", [BETWEEN], DIST, NAMES)
    assert description_rejects(
        "Go between the stone column and the grey sofa and stop about two meters before the fridge.",
        [BETWEEN], DIST, NAMES) == []
    assert "stop:after" in description_rejects(
        "Go left of the coffee table, then right of the grey sofa, and stop.", [PASS_L, PASS_R], AFTER, NAMES)


def test_between_and_through_words_required():
    assert "between:missing:0" in description_rejects(
        "Go past the stone column and the grey sofa and stop about two metres before the fridge.",
        [BETWEEN], DIST, NAMES)
    assert "through:missing:0" in description_rejects(
        "Head to the open doorway and stop in front of the fridge.", [THROUGH], NEAR, NAMES)


@pytest.mark.parametrize("name,kind,need,reason", [
    ("big wooden cabinet with doors", "object", "object", "name:long"),
    ("object", "object", "object", "name:generic"),
    ("left cabinet", "object", "object", "name:side"),
    ("outlined region", "object", "object", "name:drawing"),
    ("open doorway", "opening", "object", "kind:mismatch"),
    ("grey sofa", "object", "opening", "kind:mismatch"),
    ("", "object", "object", "name:empty"),
])
def test_name_rejects(name, kind, need, reason):
    assert reason in name_rejects(name, kind, need)


def test_good_names_pass():
    assert name_rejects("grey sofa", "object", "object") == []
    assert name_rejects("white wall", "wall", "object") == []
    assert name_rejects("arched doorway", "opening", "opening") == []


@pytest.mark.parametrize("name,plural", [
    ("bookshelf", "bookshelves"), ("shelf", "shelves"), ("knife", "knives"),
    ("pantry", "pantries"), ("table", "tables"), ("box", "boxes"),
])
def test_irregular_plurals_match_name(name, plural):
    names = {"f:x": name, "f:b5": "fridge"}
    ev = [{"type": "pass", "side": "left", "objects": ["f:x"]}]
    t = f"Go left of the {plural} and stop in front of the fridge."
    assert description_rejects(t, ev, NEAR, names) == [], t


import io  # noqa: E402

from PIL import Image  # noqa: E402

from rover_vlm.descriptions import (  # noqa: E402
    choose_shape, compose_prompt, label_object, label_route, naming_image, naming_prompt,
    parse_compose_reply, parse_name_reply,
)


def _reply(obj):
    import json
    return f"<think>ok</think>\n<answer>\n{json.dumps(obj)}\n</answer>"


def test_parse_name_reply_normalises_kind_and_name():
    assert parse_name_reply(_reply({"name": " Grey  Sofa ", "kind": "Object"})) == {"name": "grey sofa", "kind": "object"}
    assert parse_name_reply(_reply({"name": "doorway", "kind": "doorway"}))["kind"] == "opening"
    assert parse_name_reply(_reply({"name": "white wall panel"}))["name"] == "white wall"
    assert parse_name_reply(_reply({"name": "lamp", "kind": "gizmo", "extra": 1}))["kind"] == "object"
    assert parse_name_reply(_reply({"kind": "object"})) is None
    assert parse_name_reply("no answer here") is None


def test_parse_compose_reply():
    assert parse_compose_reply(_reply({"description": " Go  left of the sofa. "})) == "Go left of the sofa."
    assert parse_compose_reply("<answer>Go left of the sofa and stop.</answer>") == "Go left of the sofa and stop."
    assert parse_compose_reply(_reply({"other": 1})) is None


def test_naming_image_is_two_panels(tmp_path):
    p = tmp_path / "f.png"
    Image.new("RGB", (512, 512), (120, 120, 120)).save(p)
    img = Image.open(io.BytesIO(naming_image(p, [[0.2, 0.2], [0.6, 0.2], [0.6, 0.7]], [0.15, 0.15, 0.65, 0.75])))
    assert img.size == (1024, 512)


def test_prompts_mention_what_matters():
    assert "opening" in naming_prompt("opening") and "close-up" in naming_prompt("object")
    p = compose_prompt([PASS_L, THROUGH], NEAR, NAMES, "a polite request", "one sentence")
    assert "go left of the coffee table" in p and "go through the open doorway" in p
    assert "stop in front of the fridge" in p and "a polite request" in p
    assert choose_shape("x_L") == choose_shape("x_L")


def test_label_object_retries_once_then_rejects():
    calls = []

    def ask(img, prompt, seed):
        calls.append(seed)
        return _reply({"name": "outlined object", "kind": "object"}), "stop"
    obj = {"object_id": "f:b1", "kind_hint": "object", "image": None, "polygon": [], "crop_box": None}
    row = label_object(ask, obj, seed=3, image=b"jpeg")
    assert calls == [3, 4] and row["passed"] is False and row["name"] is None and row["rejects"]


def test_label_object_accepts_a_good_name():
    obj = {"object_id": "f:gap4", "kind_hint": "opening", "image": None, "polygon": [], "crop_box": None}
    row = label_object(lambda i, p, s: (_reply({"name": "open doorway", "kind": "opening"}), "stop"), obj, seed=0, image=b"x")
    assert row["passed"] and row["name"] == "open doorway" and row["attempts"] == 1


def test_label_route_cosmos_then_template_fallback():
    row = {"id": "f_L", "pair_id": None, "kind": "instructed", "events": [PASS_L], "stop": NEAR}
    good = "Please keep left of the coffee table and stop in front of the fridge."
    out = label_route(lambda p, s: (_reply({"description": good}), "stop"), row, NAMES, seed=0)
    assert out["passed"] and out["instruction_source"] == "cosmos" and out["description"] == good
    bad = "Go right of the coffee table and stop in front of the fridge."
    out = label_route(lambda p, s: (_reply({"description": bad}), "stop"), row, NAMES, seed=0)
    assert out["passed"] and out["instruction_source"] == "template" and "side:missing:0" in out["cosmos_rejects"]


def test_label_route_without_names_makes_no_call():
    row = {"id": "f_L", "pair_id": None, "kind": "instructed", "events": [PASS_L], "stop": NEAR}

    def boom(p, s):
        raise AssertionError("no call expected")
    out = label_route(boom, row, {"f:b5": "fridge"}, seed=0)
    assert out["passed"] is False and out["rejects"] == ["unnamed_object"] and out["missing"] == ["f:b1"]
