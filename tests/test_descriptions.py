"""Prompt d1: templates, text rules, name rules, Cosmos reply handling, finalize."""
import pytest

from rover_vlm.descriptions import (
    description_rejects, name_rejects, template_description,
)

NAMES = {"f:b1": "coffee table", "f:b2": "stone column", "f:b3": "grey sofa",
         "f:gap4": "open doorway", "f:b5": "fridge", "f:b6": "table", "f:w8": "white wall",
         "f:w9": "glass partition"}
PASS_L = {"type": "pass", "side": "left", "objects": ["f:b1"]}
PASS_R = {"type": "pass", "side": "right", "objects": ["f:b3"]}
BETWEEN = {"type": "between", "side": None, "objects": ["f:b2", "f:b3"]}
THROUGH = {"type": "through", "side": None, "objects": ["f:gap4"]}
NEAR = {"type": "near", "relation": "in front of", "distance_m": None, "object": "f:b5"}
AFTER = {"type": "after_last", "relation": "just after", "distance_m": None, "object": None}
DIST = {"type": "distance", "relation": "before", "amount": "a little", "distance_m": 2.2, "object": "f:b5"}
WALL_L = {"type": "wall", "side": "left", "objects": ["f:w8"]}
WALL_R = {"type": "wall", "side": "right", "objects": ["f:w9"]}
PASS2 = {"type": "pass", "side": "left", "objects": ["f:b1", "f:b3"]}
BETWEEN_W = {"type": "between", "side": None, "objects": ["f:b3", "f:w8"]}


@pytest.mark.parametrize("events,stop", [
    ([PASS_L], NEAR), ([PASS_L, THROUGH], NEAR), ([BETWEEN], DIST), ([PASS_L, PASS_R], AFTER), ([], NEAR),
])
def test_template_always_passes_its_own_rules(events, stop):
    text = template_description(events, stop, NAMES)
    assert description_rejects(text, events, stop, NAMES) == [], text


def test_template_wording():
    assert template_description([PASS_L, THROUGH], NEAR, NAMES) == (
        "Go left of the coffee table, then go through the open doorway, and stop in front of the fridge.")
    assert template_description([BETWEEN], DIST, NAMES).endswith("stop a little before the fridge.")


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
    assert "stop:amount" in description_rejects(
        "Go between the stone column and the grey sofa and stop before the fridge.", [BETWEEN], DIST, NAMES)
    assert description_rejects(
        "Go between the stone column and the grey sofa and stop a bit before the fridge.",
        [BETWEEN], DIST, NAMES) == []
    assert "stop:after" in description_rejects(
        "Go left of the coffee table, then right of the grey sofa, and stop.", [PASS_L, PASS_R], AFTER, NAMES)


def test_between_and_through_words_required():
    assert "between:missing:0" in description_rejects(
        "Go past the stone column and the grey sofa and stop a little before the fridge.",
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
    assert img.size == (1024, 1024)     # letterboxed square (final review I1)


def test_naming_image_letterboxes_panels_into_a_black_square(tmp_path):
    p = tmp_path / "f.png"
    Image.new("RGB", (512, 512), (120, 120, 120)).save(p)
    img = Image.open(io.BytesIO(naming_image(p, [], [0, 0, 1, 1]))).convert("RGB")
    assert img.size == (1024, 1024)
    assert sum(img.getpixel((300, 100))) < 30 and sum(img.getpixel((300, 1000))) < 30   # black bands
    assert all(abs(c - 120) < 8 for c in img.getpixel((100, 512)))                     # left panel, middle band
    assert all(abs(c - 120) < 8 for c in img.getpixel((800, 512)))                     # right panel


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


def test_parse_name_reply_cleans_punctuation_quotes_articles():
    assert parse_name_reply(_reply({"name": "Sofa."}))["name"] == "sofa"
    assert parse_name_reply(_reply({"name": '"Grey Sofa"'}))["name"] == "grey sofa"
    assert parse_name_reply(_reply({"name": "The kitchen island"}))["name"] == "kitchen island"
    assert parse_name_reply(_reply({"name": "`a lamp`!"}))["name"] == "lamp"


def test_name_rejects_flags_punctuation():
    assert "name:punct" in name_rejects("grey sofa, large", "object", "object")
    assert "name:punct" in name_rejects("sofa/table", "object", "object")
    assert "name:punct" not in name_rejects("grey sofa", "object", "object")


def test_naming_image_enlarges_small_crop(tmp_path):
    p = tmp_path / "f.png"
    Image.new("RGB", (512, 512), (200, 200, 200)).save(p)
    img = Image.open(io.BytesIO(naming_image(p, [], [0.4, 0.4, 0.5, 0.5]))).convert("L")
    right = img.crop((512, 256, 1024, 768)).point(lambda v: 255 if v > 30 else 0)
    x0, y0, x1, y1 = right.getbbox()
    assert max(x1 - x0, y1 - y0) >= 500


def test_failure_rows_keep_both_raw_replies():
    obj = {"object_id": "f:b1", "kind_hint": "object", "image": None, "polygon": [], "crop_box": None}
    row = label_object(lambda i, p, s: (f"junk{s}", "stop"), obj, seed=0, image=b"x")
    assert row["raw"] == ["junk0", "junk1"]
    r = {"id": "f_L", "pair_id": None, "kind": "instructed", "events": [PASS_L], "stop": NEAR}
    out = label_route(lambda p, s: (f"junk{s}", "stop"), r, NAMES, seed=0)
    assert out["raw"] == ["junk0", "junk1"]


from rover_vlm.descriptions import d1_gate_failures, select_routes, summarize_d1  # noqa: E402


def test_select_routes_and_gate():
    rows = [{"id": "a", "reason": None}, {"id": "b", "reason": "no_stop_anchor"}, {"id": "c", "reason": None}]
    assert [r["id"] for r in select_routes(rows)] == ["a", "c"]
    assert [r["id"] for r in select_routes(rows, limit=1)] == ["a"]
    comp = [{"id": str(i), "passed": True, "instruction_source": "cosmos", "rejects": []} for i in range(8)]
    comp += [{"id": "t", "passed": True, "instruction_source": "template", "rejects": []},
             {"id": "e", "error": "Timeout"}]
    s = summarize_d1(comp, "compose")
    assert s["n"] == 10 and s["errors"] == 1 and s["accept_rate"] == 1.0
    assert s["template_share"] == pytest.approx(1 / 9, abs=1e-3)
    assert d1_gate_failures(s, "compose") == ["template share 0.111 > 0.1", "request errors 1 > 0"]
    names = [{"id": "o1", "passed": True, "rejects": []}, {"id": "o2", "passed": False, "rejects": ["name:long"]}]
    assert d1_gate_failures(summarize_d1(names, "name"), "name") == ["accept rate 0.500 < 0.9"]


def test_name_rows_split_and_missing():
    from rover_vlm.descriptions import name_rows_split, names_missing
    rows = [{"id": "a", "passed": True, "name": "sofa"}, {"id": "b", "passed": False, "rejects": ["parse"]},
            {"id": "c", "error": "Timeout"}]
    named, rejected = name_rows_split(rows)
    assert named == {"a": "sofa"} and rejected == {"b"}
    route = {"events": [{"objects": ["a", "b"]}, {"objects": ["c"]}], "stop": {"object": "d"}}
    assert names_missing(route, named, rejected) == ["c", "d"]
    assert names_missing({"events": [{"objects": ["a", "b"]}], "stop": {"object": None}}, named, rejected) == []


def test_gate_fails_on_request_errors():
    s = {"n": 10, "errors": 2, "accept_rate": 1.0, "rejects": {}}
    assert d1_gate_failures(s, "name") == ["request errors 2 > 0"]


from rover_vlm.habitat_data import description_entry_prompt, instruction_endpoints_prompt  # noqa: E402


def test_entry_prompt_has_the_entry_and_text_but_no_goal():
    p = description_entry_prompt([0.167, 1.0, 1], 'Go left of the "sofa" and stop by the fridge.')
    assert p.startswith("<image>\n") and "[0.167, 1.0]" in p and "where it is visible" in p
    assert "Go left of the 'sofa' and stop by the fridge." in p
    assert "goal" not in p.lower()
    q = instruction_endpoints_prompt([0.167, 1.0, 1], [0.5, 0.6, 1], "x")
    assert p.split("Coordinates are")[1].split("The first waypoint")[0] == q.split("Coordinates are")[1].split("The first waypoint")[0].replace(
        "Predict the traversable route between those two points", "Predict the route")


import copy  # noqa: E402
import json as _json  # noqa: E402

from rover_vlm.descriptions import (  # noqa: E402
    desc_check, finalize_descriptions, prompt_form, shuffled_texts, with_description,
)


def _rec(rid, sample, pair=None):
    path = [[0.2, 1.0, 1], [0.3, 0.8, 1], [0.5, 0.6, 1]]
    return {"id": rid, "image": ["x.png"],
            "conversations": [{"from": "human", "value": "old"}, {"from": "gpt", "value": _json.dumps({"path": path})}],
            "habitat_meta": {"sample_id": sample, "pair_id": pair, "scene_id": "s", "kind": "instructed"}}


def _ev(rid):
    return {"id": rid, "reason": None, "events": [PASS_L], "stop": NEAR, "goal_xz": [1.0, 2.0],
            "pose_xz": [0.0, 0.0], "heading_xz": [0.0, 1.0]}


OBJS = {"f:b1": {"cells_xz": [[0.1, 1.0], [0.2, 1.0]]}, "f:b5": {"cells_xz": [[1.0, 2.5]]}}


def test_prompt_form_is_stable_and_about_seventy_percent():
    forms = [prompt_form(f"frame{i}") for i in range(2000)]
    assert forms == [prompt_form(f"frame{i}") for i in range(2000)]
    assert 0.66 < forms.count("entry") / len(forms) < 0.74


def test_with_description_forms_and_check():
    chk = desc_check(_ev("a"), OBJS)
    assert chk["events"][0]["obstacles"] == [[[0.1, 1.0], [0.2, 1.0]]] and chk["goal_xz"] == [1.0, 2.0]
    rec = _rec("a", "f1")
    e = with_description(rec, "Go left of the table and stop.", "cosmos", "entry", chk)
    g = with_description(rec, "Go left of the table and stop.", "cosmos", "goal", chk)
    assert "goal" not in e["conversations"][0]["value"].lower() and "[0.5, 0.6]" in g["conversations"][0]["value"]
    assert e["habitat_meta"]["goal_in_prompt"] is False and g["habitat_meta"]["goal_in_prompt"] is True
    assert rec["conversations"][0]["value"] == "old"                      # not mutated


def test_finalize_keeps_survivors_and_only_full_pairs():
    recs = [_rec("p_L", "f1", "p"), _rec("p_R", "f1", "p"), _rec("q_L", "f2", "q"), _rec("q_R", "f2", "q"),
            _rec("s_L", "f3")]
    events = {r["id"]: _ev(r["id"]) for r in recs}
    events["q_R"] = dict(events["q_R"], reason="no_stop_anchor")
    desc = {r["id"]: {"id": r["id"], "passed": True, "description": "Go left of the table and stop.",
                      "instruction_source": "cosmos"} for r in recs}
    out = finalize_descriptions(recs, events, desc, OBJS)
    assert [r["id"] for r in out["kept"]] == ["p_L", "p_R", "q_L", "s_L"]
    assert [r["id"] for r in out["pairs"]] == ["p_L", "p_R"]
    forms = {r["id"]: r["habitat_meta"]["prompt_form"] for r in out["kept"]}
    assert forms["p_L"] == forms["p_R"]                                   # a pair shares its form
    assert out["stats"]["dropped_geometry"] == 1 and out["stats"]["pairs_kept"] == 1


def test_shuffled_texts_come_from_other_frames():
    recs = []
    for i in range(6):
        r = with_description(_rec(f"r{i}", f"f{i // 2}"), f"text {i}", "cosmos", "entry", desc_check(_ev("x"), OBJS))
        recs.append(r)
    out = shuffled_texts(recs)
    by = {r["id"]: r for r in recs}
    for r in out:
        src = next(o for o in recs if o["habitat_meta"]["instruction"] == r["habitat_meta"]["instruction"])
        assert src["habitat_meta"]["sample_id"] != by[r["id"]]["habitat_meta"]["sample_id"]
        assert r["habitat_meta"]["instruction_source"] == "shuffled"
        assert r["habitat_meta"]["desc_check"] == by[r["id"]]["habitat_meta"]["desc_check"]


# --- final-review fixes (F1-F4) -----------------------------------------------------------
from rover_vlm.route_events import AMOUNT_STEPS  # noqa: E402

SOFA_PAST = {"type": "distance", "relation": "past", "amount": "a little", "distance_m": 2.0, "object": "f:b3"}
NAMES_F = dict(NAMES, **{"f:b7": "armchair"})


def test_pronoun_stop_accepted_when_stop_object_is_the_last_events():
    t = "Go right of the grey sofa, then stop a little past it."
    assert description_rejects(t, [PASS_R], SOFA_PAST, NAMES) == []
    near = {"type": "near", "relation": "just past", "amount": None, "distance_m": None, "object": "f:b3"}
    assert description_rejects("Go right of the grey sofa and stop just past them.", [PASS_R], near, NAMES) == []


def test_pronoun_stop_rejected_for_a_new_object():
    t = "Go left of the coffee table, then stop a little before it."
    assert "stop:name" in description_rejects(t, [PASS_L], DIST, NAMES)


def test_template_keeps_the_explicit_stop_name():
    t = template_description([PASS_R], SOFA_PAST, NAMES)
    assert t.endswith("stop a little past the grey sofa.")
    assert description_rejects(t, [PASS_R], SOFA_PAST, NAMES) == []


def test_right_in_stop_phrases_is_not_a_side_word():
    ev = [PASS_L, PASS_R]
    assert description_rejects("Go left of the coffee table, then right of the grey sofa, and stop right after it.",
                               ev, AFTER, NAMES) == []
    near = {"type": "near", "relation": "in front of", "distance_m": None, "object": "f:b7"}
    assert description_rejects("Go left of the coffee table and stop right in front of the armchair.",
                               [PASS_L], near, NAMES_F) == []
    jp = {"type": "near", "relation": "just past", "distance_m": None, "object": "f:b7"}
    assert description_rejects("Go left of the coffee table and stop right past the armchair.",
                               [PASS_L], jp, NAMES_F) == []


def test_real_side_word_still_binds():
    # "go right of" still counts; only the "right in front of" in the stop clause is neutral
    assert "side:missing:0" in description_rejects(
        "Go right of the coffee table and stop right in front of the fridge.", [PASS_L], NEAR, NAMES)
    assert description_rejects(
        "Go right of the grey sofa and stop right in front of the fridge.", [PASS_R], NEAR, NAMES) == []
    assert "side:stray" in description_rejects(
        "Go right of the grey sofa and stop in front of the fridge on the right.", [PASS_R], NEAR, NAMES)


@pytest.mark.parametrize("name", ["table and chairs", "sofa or couch", "shelf then wall"])
def test_name_rejects_conjunctions(name):
    assert "name:conjunction" in name_rejects(name, "object", "object")


@pytest.mark.parametrize("name", ["furniture", "piece of furniture", "obstacle", "wooden piece"])
def test_name_rejects_more_generic_names(name):
    assert "name:generic" in name_rejects(name, "object", "object")


# --- Task 1: name rules N1 + N2 (rule for doors in walls, unit modifiers) ---

def test_name_rule_n1_opening_for_wall_not_mismatch():
    # N1: kind "opening" for need "wall" is NOT a kind:mismatch (doors/windows in a wall)
    assert name_rejects("open doorway", "opening", "wall") == []
    assert name_rejects("glass door", "opening", "wall") == []
    assert name_rejects("window with blind", "opening", "wall") == []


def test_name_rule_n1_object_for_opening_is_mismatch():
    # N1: need "opening" + kind "object" stays a mismatch
    assert "kind:mismatch" in name_rejects("wooden frame", "object", "opening")


def test_name_rule_n1_object_plus_opening_kind_is_mismatch():
    # N1: need "object" + kind "opening" stays a mismatch
    assert "kind:mismatch" in name_rejects("open doorway", "opening", "object")


def test_name_rule_n2_unit_with_modifier_not_generic():
    # N2: "unit" is not generic when preceded by a modifier from UNIT_MODIFIERS
    assert name_rejects("white shelving unit", "object", "object") == []
    assert name_rejects("shelf unit", "object", "object") == []
    assert name_rejects("storage unit", "object", "object") == []
    assert name_rejects("kitchen unit", "object", "object") == []
    assert name_rejects("wall unit", "object", "object") == []
    assert name_rejects("tv unit", "object", "object") == []
    assert name_rejects("display unit", "object", "object") == []
    assert name_rejects("corner unit", "object", "object") == []
    assert name_rejects("drawer unit", "object", "object") == []
    assert name_rejects("cabinet unit", "object", "object") == []
    # Trailing phrases should not interfere with the modifier check
    assert name_rejects("shelving unit with books", "object", "object") == []
    assert name_rejects("white shelving unit", "object", "object") == []
    assert "name:generic" not in name_rejects("white shelf unit with door", "object", "object")
    assert name_rejects("tv unit on wall", "object", "object") == []


def test_name_rule_n2_unit_without_modifier_is_generic():
    # N2: "unit" stays generic without a modifier
    assert "name:generic" in name_rejects("unit", "object", "object")
    assert "name:generic" in name_rejects("white unit", "object", "object")
    assert "name:generic" in name_rejects("large unit", "object", "object")
    assert "name:generic" in name_rejects("blue unit", "object", "object")


@pytest.mark.parametrize("text,stop", [
    ("Go left of the coffee table and stop a little before the fridge.",
     {"type": "distance", "relation": "past", "amount": "a little", "distance_m": 2.0, "object": "f:b5"}),
    ("Go left of the coffee table and stop a little past the fridge.", DIST),
    ("Go left of the coffee table and stop just past the fridge.", NEAR),
    ("Go left of the coffee table and stop next to the fridge.", NEAR),
    ("Go left of the coffee table and stop in front of the fridge.",
     {"type": "near", "relation": "just past", "distance_m": None, "object": "f:b5"}),
])
def test_swapped_stop_relation_rejected(text, stop):
    assert "stop:relation" in description_rejects(text, [PASS_L], stop, NAMES)


@pytest.mark.parametrize("text,stop", [
    ("Go left of the coffee table and stop just in front of the fridge.", NEAR),
    ("Go left of the coffee table and stop just beyond the fridge.",
     {"type": "near", "relation": "just past", "distance_m": None, "object": "f:b5"}),
    ("Go left of the coffee table and stop a little short of the fridge.", DIST),
    ("Go left of the coffee table and stop slightly beyond the fridge.",
     {"type": "distance", "relation": "past", "amount": "a little", "distance_m": 2.0, "object": "f:b5"}),
])
def test_stop_relation_synonyms_pass(text, stop):
    assert description_rejects(text, [PASS_L], stop, NAMES) == [], text


_AMOUNTS = [w for _, w in AMOUNT_STEPS]
_ALL_STOPS = ([{"type": "near", "relation": r, "amount": None, "distance_m": None, "object": o}
               for r in ("in front of", "just past") for o in ("f:b5", "f:b3")]
              + [{"type": "near", "relation": "in front of", "amount": None, "distance_m": None, "object": "f:w8"}]
              + [{"type": "distance", "relation": r, "amount": a, "distance_m": 2.0, "object": o}
                 for r in ("before", "past") for a in _AMOUNTS for o in ("f:b5", "f:b3", "f:w8")]
              + [AFTER])
_ALL_EVENTS = [[PASS_R], [PASS2], [WALL_L], [WALL_R], [BETWEEN], [BETWEEN_W], [THROUGH],
               [PASS_L, PASS_R], [THROUGH, PASS_R], [WALL_R, PASS2], [PASS_R, WALL_L], [WALL_L, WALL_R], []]


@pytest.mark.parametrize("stop", _ALL_STOPS, ids=lambda s: f"{s['type']}-{s['relation']}-{s.get('amount')}-{s['object']}")
@pytest.mark.parametrize("events", _ALL_EVENTS, ids=lambda ev: "+".join(f"{e['type']}{len(e['objects'])}{e['side'] or ''}"
                                                                          for e in ev) or "none")
def test_template_passes_for_every_stop(events, stop):
    for kinds in (None, {"f:w8": "wall", "f:w9": "wall"}):
        text = template_description(events, stop, NAMES, kinds=kinds)
        assert description_rejects(text, events, stop, NAMES) == [], text


# --- Addendum A: walls, multi-object passes, coarse stop amounts (Task 15) -----------------
from rover_vlm.descriptions import stop_text  # noqa: E402

W_NEAR = {"type": "near", "relation": "in front of", "amount": None, "distance_m": None, "object": "f:b5"}


def test_new_event_templates():
    assert template_description([PASS2], W_NEAR, NAMES) == (
        "Go left of the coffee table and the grey sofa, and stop in front of the fridge.")
    assert template_description([WALL_L, PASS_R], W_NEAR, NAMES) == (
        "Keep the white wall on your left, then go right of the grey sofa, and stop in front of the fridge.")
    assert template_description([BETWEEN_W], W_NEAR, NAMES).startswith("Go between the grey sofa and the white wall,")


def test_distance_stop_templates_use_amounts():
    well = {"type": "distance", "relation": "past", "amount": "well", "distance_m": 4.2, "object": "f:b3"}
    assert stop_text(well, "grey sofa") == "stop well past the grey sofa"
    just = {"type": "distance", "relation": "before", "amount": "just", "distance_m": 0.9, "object": "f:b5"}
    assert stop_text(just, "fridge") == "stop just before the fridge"


def test_wall_past_stop_says_the_end_of_the_wall():
    st = {"type": "distance", "relation": "past", "amount": "well", "distance_m": 4.0, "object": "f:w8"}
    t = template_description([WALL_L], st, NAMES)                           # wall-ness from the wall event
    assert t.endswith("stop well past the end of the white wall.")
    t = template_description([BETWEEN_W], st, NAMES, kinds={"f:w8": "wall"})  # from kinds (inside between)
    assert t.endswith("stop well past the end of the white wall.")
    for text in ("Keep the white wall on your left and stop well past the end of the white wall.",
                 "Keep the white wall on your left and stop well past the white wall.",
                 "Keep the white wall on your left and stop far past it."):
        assert description_rejects(text, [WALL_L], st, NAMES) == [], text
    before = dict(st, relation="before")
    assert template_description([], before, NAMES, kinds={"f:w8": "wall"}).endswith("stop well before the white wall.")


def test_near_in_front_of_wall():
    st = {"type": "near", "relation": "in front of", "amount": None, "distance_m": None, "object": "f:w8"}
    t = template_description([PASS_R], st, NAMES, kinds={"f:w8": "wall"})
    assert t.endswith("stop in front of the white wall.") and description_rejects(t, [PASS_R], st, NAMES) == []


def test_distance_stop_without_amount_raises():
    old = {"type": "distance", "relation": "past", "distance_m": 2.0, "object": "f:b3"}
    with pytest.raises(ValueError, match="distance stop without amount"):
        stop_text(old, "grey sofa")
    with pytest.raises(ValueError, match="distance stop without amount"):
        template_description([PASS_R], dict(old, amount=None), NAMES)


@pytest.mark.parametrize("amount,ok,bad", [
    ("just", ["just", "right"], ["well", "a little", ""]),
    ("a little", ["a little", "a bit", "slightly", "shortly"], ["well", "just", ""]),
    ("well", ["well", "far", "a good way"], ["a little", "just", ""]),
])
def test_amount_words(amount, ok, bad):
    st = {"type": "distance", "relation": "past", "amount": amount, "distance_m": 2.0, "object": "f:b5"}
    for w in ok:
        t = f"Go left of the coffee table and stop {w} past the fridge."
        assert description_rejects(t, [PASS_L], st, NAMES) == [], t
    for w in bad:
        t = " ".join(f"Go left of the coffee table and stop {w} past the fridge.".split())
        assert "stop:amount" in description_rejects(t, [PASS_L], st, NAMES), t


def test_amount_word_must_be_in_the_stop_clause():
    st = {"type": "distance", "relation": "past", "amount": "well", "distance_m": 4.0, "object": "f:b5"}
    t = "Go well left of the coffee table and stop past the fridge."
    assert "stop:amount" in description_rejects(t, [PASS_L], st, NAMES)


def test_next_to_is_no_longer_a_relation():
    from rover_vlm.descriptions import STOP_RELATION_WORDS
    assert ("near", "next to") not in STOP_RELATION_WORDS


@pytest.mark.parametrize("text", [
    "Pass the coffee table and the grey sofa on the left, and stop in front of the fridge.",
    "Go left of the coffee table and the grey sofa, then stop in front of the fridge.",
    "Keep left of the coffee table and the grey sofa and stop in front of the fridge.",
])
def test_multi_object_pass_accepts_one_side_word(text):
    assert description_rejects(text, [PASS2], W_NEAR, NAMES) == [], text


@pytest.mark.parametrize("text,reason", [
    ("Go left of the coffee table and right of the grey sofa, then stop in front of the fridge.", "side:opposite:0"),
    ("Go past the coffee table and the grey sofa, then stop in front of the fridge.", "side:missing:0"),
    ("Go left of the coffee table, then stop in front of the fridge.", "name:missing:0"),
])
def test_multi_object_pass_rejections(text, reason):
    assert reason in description_rejects(text, [PASS2], W_NEAR, NAMES)


@pytest.mark.parametrize("text", [
    "Keep the white wall on your left, then go right of the grey sofa, and stop in front of the fridge.",
    "Follow the white wall on your left, go right of the grey sofa and stop in front of the fridge.",
    "With the white wall to your left, go right of the grey sofa and stop in front of the fridge.",
])
def test_wall_event_side_word_accepted(text):
    assert description_rejects(text, [WALL_L, PASS_R], W_NEAR, NAMES) == [], text


@pytest.mark.parametrize("text,reason", [
    ("Keep the white wall on your right, then go right of the grey sofa, and stop in front of the fridge.",
     "side:opposite:0"),
    ("Keep the white wall on your right, then go right of the grey sofa, and stop in front of the fridge.",
     "side:egocentric"),
    ("Follow the white wall, then go right of the grey sofa, and stop in front of the fridge.", "side:missing:0"),
    # a wall's side word counts only in egocentric form: "on the left" / "left of" / "keep left"
    # read as the ROUTE's side for objects, the opposite geometry (fix round 1, M1)
    ("Keep left of the white wall, then go right of the grey sofa, and stop in front of the fridge.",
     "side:wall_form:0"),
    ("Keep left of the white wall, then go right of the grey sofa, and stop in front of the fridge.",
     "side:missing:0"),
    ("Pass the white wall on the left, then go right of the grey sofa, and stop in front of the fridge.",
     "side:wall_form:0"),
    ("Pass the white wall on the left, then go right of the grey sofa, and stop in front of the fridge.",
     "side:missing:0"),
    ("Keep the white wall to the left, then go right of the grey sofa, and stop in front of the fridge.",
     "side:wall_form:0"),
    ("Keep the white wall on your left, on the left, then go right of the grey sofa, and stop in front of "
     "the fridge.", "side:wall_form:0"),
])
def test_wall_event_rejections(text, reason):
    assert reason in description_rejects(text, [WALL_L, PASS_R], W_NEAR, NAMES)


@pytest.mark.parametrize("text", [
    # egocentric next to an OBJECT name is still rejected (Review Focus 4)
    "Keep the white wall on your left, then pass the grey sofa on your right, and stop in front of the fridge.",
    "Keep the white wall on your left, then go right of the grey sofa, and stop in front of the fridge on your left.",
])
def test_egocentric_next_to_an_object_still_rejected(text):
    assert "side:egocentric" in description_rejects(text, [WALL_L, PASS_R], W_NEAR, NAMES)


def test_wall_inside_between_takes_no_side_word():
    t = "Go between the grey sofa and the white wall and stop in front of the fridge."
    assert description_rejects(t, [BETWEEN_W], W_NEAR, NAMES) == []
    t = "Go between the grey sofa and the white wall on your left and stop in front of the fridge."
    assert "side:egocentric" in description_rejects(t, [BETWEEN_W], W_NEAR, NAMES)


def test_compose_prompt_wall_language():
    p = compose_prompt([WALL_L, PASS2], W_NEAR, NAMES, "a polite request", "one sentence")
    assert "1. keep the white wall on your left" in p
    assert "2. go left of the coffee table and the grey sofa" in p
    assert 'Never say "your left" or "your right".' not in p


def test_naming_prompt_and_name_rules_for_walls():
    # final review I3: "wall" is any long blob the route runs alongside or stops in front of
    p = " ".join(naming_prompt("wall").split())
    assert ("something long that the rover drives alongside or stops in front of: a wall, a partition, a door in "
            "a wall, or a long piece of furniture such as a sofa or a counter.") in p
    assert "part of a wall or a door in a wall that the rover drives alongside" not in p
    assert name_rejects("white wall", "wall", "wall") == []
    assert name_rejects("grey sofa", "object", "wall") == []
    assert name_rejects("wooden door", "object", "wall") == []


def test_label_object_asks_with_the_wall_prompt():
    seen = []
    obj = {"object_id": "f:w8", "kind_hint": "wall", "image": None, "polygon": [], "crop_box": None}

    def ask(img, prompt, seed):
        seen.append(prompt)
        return _reply({"name": "white wall", "kind": "wall"}), "stop"
    row = label_object(ask, obj, seed=0, image=b"x")
    assert row["passed"] and row["need"] == "wall" and "drives alongside" in seen[0]


def test_label_route_wall_template_fallback_uses_kinds():
    st = {"type": "distance", "relation": "past", "amount": "well", "distance_m": 4.0, "object": "f:w8"}
    row = {"id": "f_L", "pair_id": None, "kind": "instructed", "events": [BETWEEN_W], "stop": st}
    out = label_route(lambda p, s: (_reply({"description": "junk"}), "stop"), row, NAMES, seed=0,
                      kinds={"f:w8": "wall"})
    assert out["passed"] and out["instruction_source"] == "template"
    assert out["description"].endswith("stop well past the end of the white wall.")


def test_desc_check_carries_wall_and_multi_object_pass():
    objs = dict(OBJS, **{"f:b3": {"cells_xz": [[-1.0, 2.0]]}, "f:w8": {"cells_xz": [[2.0, 1.0], [2.0, 3.0]]}})
    ev = dict(_ev("a"), events=[WALL_L, PASS2])
    chk = desc_check(ev, objs)
    assert chk["events"][0] == {"type": "wall", "side": "left", "obstacles": [[[2.0, 1.0], [2.0, 3.0]]],
                                "gap_xz": None}
    assert chk["events"][1]["type"] == "pass" and chk["events"][1]["obstacles"] == [
        [[0.1, 1.0], [0.2, 1.0]], [[-1.0, 2.0]]]


def test_wall_egocentric_form_survives_the_right_adverb_blanking():
    # "on your right past ..." : the right-adverb blanking must not hide the wall's egocentric word
    t = "Follow the glass partition on your right past its end, and stop just after it."
    assert description_rejects(t, [WALL_R], AFTER, NAMES) == []


def test_compose_prompt_d2_direction_paragraph():
    from rover_vlm.descriptions import PROMPT_VERSION
    p = " ".join(compose_prompt([WALL_L, PASS2], W_NEAR, NAMES, "a polite request", "one sentence").split())
    assert PROMPT_VERSION == "d3"
    assert (
        "Keep every step in this order and keep each object's name. Copy each step's direction words exactly as written: in a 'keep the ... on your left/right' step, 'on your left' / 'on your right' stays that way, whatever the thing is (a wall, a sofa, a counter); in a 'go left of / right of' step, 'left of' / 'right of' stays that way; a step with 'between' or 'through' has no left or right, so do not add one. Keep where to stop."
    ) in p
    assert "after a wall stays that way" not in p
    assert "the word left or right that goes with it" not in p
    assert "never say 'your left/right'" not in p


def test_label_descriptions_max_tokens_default_is_4096():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "label_descriptions.py"
    spec = importlib.util.spec_from_file_location("label_descriptions_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DEFAULT_MAX_TOKENS == 4096


WHOLE_THING = ("Name the whole thing the outline covers, not a small part on it (a handle, a switch, "
               "a socket, a leg).")


def test_naming_prompt_says_whole_thing_for_every_need():
    from rover_vlm.descriptions import naming_prompt
    for need in ("opening", "wall", "object", "anything"):
        p = " ".join(naming_prompt(need).split())
        assert WHOLE_THING in p
        assert p.index("1. Name what is inside the outline") < p.index(WHOLE_THING) < p.index("2. Say what kind")


def test_d1_styles():
    from rover_vlm.descriptions import D1_STYLES
    from rover_vlm.instructions import STYLES
    assert len(D1_STYLES) == 4
    assert "a command that also says briefly why or where it leads" not in D1_STYLES
    assert "a calm, step-by-step instruction" in D1_STYLES
    assert STYLES[-1] == "a command that also says briefly why or where it leads" and len(STYLES) == 4
    assert "a calm, step-by-step instruction" not in STYLES


def test_label_route_uses_d1_styles():
    from rover_vlm.descriptions import D1_STYLES, choose_d1_style, label_route
    assert {choose_d1_style(f"r{i}") for i in range(200)} == set(D1_STYLES)
    assert choose_d1_style("r7") == choose_d1_style("r7")
    row = {"pair_id": None, "kind": "instructed", "events": [PASS_L], "stop": NEAR}
    for rid in ("r1", "r2", "r3", "r4", "r5", "r6"):
        seen = []
        label_route(lambda p, s: (seen.append(" ".join(p.split())) or "", "stop"), {**row, "id": rid}, NAMES, seed=0)
        assert choose_d1_style(rid) in seen[0]


# ---- mixed namer: name clash + fallback names -------------------------------------------------
def _route(events, stop):
    return {"id": "f_L", "pair_id": None, "kind": "instructed", "events": events, "stop": stop}


def test_name_clash_same_head_noun_on_two_ids():
    from rover_vlm.descriptions import name_clash
    row = _route([BETWEEN], NEAR)           # f:b2, f:b3, stop f:b5
    assert not name_clash(row, NAMES)
    assert name_clash(row, {**NAMES, "f:b2": "wooden bench", "f:b3": "old bench"})
    assert name_clash(row, {**NAMES, "f:b5": "stone column"})      # event member vs stop object


def test_name_clash_same_id_twice_and_unnamed_are_not_clashes():
    from rover_vlm.descriptions import name_clash
    row = _route([PASS_L, PASS_L], NEAR)    # f:b1 twice
    assert not name_clash(row, NAMES)
    row = _route([BETWEEN], NEAR)
    assert not name_clash(row, {"f:b2": "bench"})                  # f:b3 / f:b5 unnamed


def _good_ask(text):
    return lambda p, s: (_reply({"description": text}), "stop")


GOOD = "Please keep left of the coffee table and stop in front of the fridge."


def test_label_route_primary_names_tagged_primary():
    out = label_route(_good_ask(GOOD), _route([PASS_L], NEAR), NAMES, seed=0, fallback_names={"f:b1": "x"})
    assert out["passed"] and out["namer"] == "primary"


def test_label_route_uses_fallback_when_primary_unnamed():
    primary = {"f:b5": "fridge"}            # f:b1 unnamed
    out = label_route(_good_ask(GOOD), _route([PASS_L], NEAR), primary, seed=0, fallback_names=NAMES)
    assert out["passed"] and out["namer"] == "fallback" and out["description"] == GOOD


def test_label_route_uses_fallback_on_primary_clash_and_composes_with_fallback_names():
    row = _route([BETWEEN], NEAR)
    primary = {**NAMES, "f:b2": "wooden bench", "f:b3": "wooden bench"}
    seen = []
    out = label_route(lambda p, s: (seen.append(p) or "junk", "stop"), row, primary, seed=0, fallback_names=NAMES)
    assert out["namer"] == "fallback" and any("stone column" in " ".join(p.split()) for p in seen)


def test_label_route_clash_in_both_rejects_name_clash_without_a_call():
    row = _route([BETWEEN], NEAR)
    clash = {**NAMES, "f:b2": "wooden bench", "f:b3": "wooden bench"}

    def boom(p, s):
        raise AssertionError("no call expected")
    out = label_route(boom, row, clash, seed=0, fallback_names=clash)
    assert out["passed"] is False and out["rejects"] == ["name:clash"] and out["description"] is None
    out = label_route(boom, row, clash, seed=0)       # Cosmos-only run: clash rule still applies
    assert out["rejects"] == ["name:clash"]


def test_label_route_unnamed_in_both_is_unnamed_object():
    def boom(p, s):
        raise AssertionError("no call expected")
    out = label_route(boom, _route([PASS_L], NEAR), {"f:b5": "fridge"}, seed=0, fallback_names={"f:b5": "fridge"})
    assert out["rejects"] == ["unnamed_object"] and out["missing"] == ["f:b1"]


def test_names_missing_counts_a_name_in_either_file():
    from rover_vlm.descriptions import names_missing
    route = {"events": [{"objects": ["a", "b"]}, {"objects": ["c"]}], "stop": {"object": "d"}}
    assert names_missing(route, {"a": "x"}, {"b"}, {"c": "y", "b": "z"}, set()) == ["d"]
    # rejected in primary but only an error row in fallback -> still a retryable gap
    assert names_missing(route, {"a": "x", "c": "y", "d": "w"}, {"b"}, {}, set()) == ["b"]
    assert names_missing(route, {"a": "x", "c": "y", "d": "w"}, {"b"}, {}, {"b"}) == []


def test_summarize_d1_reports_namer_counts():
    comp = [{"id": "1", "passed": True, "instruction_source": "cosmos", "namer": "primary", "rejects": []},
            {"id": "2", "passed": True, "instruction_source": "cosmos", "namer": "primary", "rejects": []},
            {"id": "3", "passed": True, "instruction_source": "cosmos", "namer": "fallback", "rejects": []},
            {"id": "4", "passed": False, "rejects": ["name:clash"]}]
    assert summarize_d1(comp, "compose")["namer"] == {"primary": 2, "fallback": 1}


# ---- I1: Cosmos names only the objects of routes unusable under the Gemma names ----------------
def _name_rows(names, bad=(), err=()):
    return ([{"id": k, "name": v, "passed": True} for k, v in names.items()]
            + [{"id": k, "name": None, "passed": False} for k in bad]
            + [{"id": k, "error": "boom"} for k in err])


def test_objects_needing_fallback_selects_clashing_and_unnamed_routes_only():
    from rover_vlm.descriptions import objects_needing_fallback
    ok = {"id": "ok", "events": [PASS_L], "stop": NEAR}                       # f:b1, f:b5
    clash = {"id": "cl", "events": [BETWEEN], "stop": NEAR}                   # f:b2, f:b3, f:b5
    unnamed = {"id": "un", "events": [PASS_R, THROUGH], "stop": {**NEAR, "object": None}}   # f:b3, f:gap4
    names = {"f:b1": "coffee table", "f:b5": "fridge", "f:b2": "wooden bench", "f:b3": "wooden bench"}
    rows = _name_rows(names, bad=["f:gap4"])                                  # gap4 rejected -> unnamed
    assert objects_needing_fallback([ok, clash, unnamed], rows) == {"f:b2", "f:b3", "f:b5", "f:gap4"}
    # an error row (or no row at all) counts as unnamed too
    rows = _name_rows({"f:b1": "coffee table"}, err=["f:b5"])
    assert objects_needing_fallback([ok], rows) == {"f:b1", "f:b5"}
    assert objects_needing_fallback([ok], _name_rows({"f:b1": "coffee table", "f:b5": "fridge"})) == set()


def test_primary_usable_route_composes_with_fallback_file_lacking_its_objects():
    from rover_vlm.descriptions import names_missing
    route = _route([PASS_L], NEAR)
    named, rejected = {"f:b1": "coffee table", "f:b5": "fridge"}, set()
    assert names_missing(route, named, rejected, {}, set()) == []   # fallback never ran these objects
    out = label_route(_good_ask(GOOD), route, named, seed=0, fallback_names={})
    assert out["passed"] and out["namer"] == "primary"
