"""Route checker (spec docs/superpowers/specs/2026-11-16-route-checker-design.md): prompt k1, reply parsing, the
per-route check call."""
import json

from rover_vlm.route_check import CHECK_PROMPT_VERSION, check_prompt, check_route, parse_check_reply

GOOD = {"names": [{"name": "grey sofa", "fits": "yes", "is_object": "yes"}], "missing_obstacle": "no",
        "stop_ok": "yes", "better_stop": "", "overall_true": "yes", "confidence": 80}


def test_prompt_states_legend_and_never_sides():
    text = "Go left of the sofa, then stop next to the lamp."
    p = check_prompt(text)
    assert text in p
    for word in ("magenta", "dashed", "green", "orange", "middle of the room"):
        assert word in p
    assert "already correct" in p
    assert CHECK_PROMPT_VERSION == "k1"


def test_parse_plain_json():
    r = parse_check_reply(json.dumps(GOOD))
    assert r == {"names": [{"name": "grey sofa", "fits": "yes", "is_object": "yes"}], "missing_obstacle": False,
                 "stop_ok": True, "better_stop": "", "overall_true": True, "confidence": 80}


def test_parse_after_think_and_code_fence():
    raw = "<think>the sofa is grey {not json}</think>```json\n" + json.dumps({**GOOD, "overall_true": False}) \
        + "\n```\nThat is my answer."
    r = parse_check_reply(raw)
    assert r is not None and r["overall_true"] is False and r["stop_ok"] is True


def test_parse_bad_returns_none():
    assert parse_check_reply("The route looks fine to me.") is None
    assert parse_check_reply(json.dumps({k: v for k, v in GOOD.items() if k != "overall_true"})) is None


def test_parse_empty_names_ok():
    r = parse_check_reply(json.dumps({**GOOD, "names": []}))
    assert r is not None and r["names"] == []


def test_confidence_forms():
    def conf(v):
        d = dict(GOOD)
        if v is None:
            d.pop("confidence")
        else:
            d["confidence"] = v
        return parse_check_reply(json.dumps(d))["confidence"]
    assert conf("85%") == 85
    assert conf(85.0) == 85
    assert conf(140) == 100
    assert conf(-3) == 0
    assert conf(None) == 50


def test_check_route_retries_then_parse_reject():
    item = {"id": "g3/00800-TEEsavR23oF_c000_L", "image": "/x.jpg", "text": "Stop by the wall."}
    seeds = []

    def prose(image, prompt, seed):
        seeds.append(seed)
        return "no json here", "stop"
    row = check_route(prose, item, 7)
    assert row["rejects"] == ["parse"] and row["passed"] is False and row["attempts"] == 2
    assert len(row["raw"]) == 2 and row["id"] == item["id"] and seeds == [7, 8]

    replies = iter(["prose first", json.dumps(GOOD)])
    seeds.clear()

    def second(image, prompt, seed):
        seeds.append(seed)
        return next(replies), "stop"
    row = check_route(second, item, 7)
    assert row["passed"] is True and row["attempts"] == 2 and seeds == [7, 8]
    assert row["overall_true"] is True and row["prompt_version"] == "k1" and row["rejects"] == []
