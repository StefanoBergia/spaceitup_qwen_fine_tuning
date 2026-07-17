from rover_vlm.habitat_data import normalize_points, clip_polyline_unit, resample_with_transitions


def test_normalize_divides_by_size():
    out = normalize_points([(256.0, 128.0, False), (512.0, 512.0, True)], 512, 512)
    assert out == [(0.5, 0.25, False), (1.0, 1.0, True)]


def test_clip_keeps_fully_inside():
    pts = [(0.2, 0.9, False), (0.3, 0.5, False), (0.4, 0.2, False)]
    assert clip_polyline_unit(pts) == pts


def test_clip_inserts_bottom_boundary():
    # path starts below the frame (y > 1) then enters; expect a point at y == 1
    pts = [(0.5, 1.6, False), (0.5, 0.4, False)]
    out = clip_polyline_unit(pts)
    assert all(0.0 <= y <= 1.0 for _, y, _ in out)
    assert any(abs(y - 1.0) < 1e-6 for _, y, _ in out)
    assert out[-1] == (0.5, 0.4, False)


def test_clip_preserves_visibility_transition_no_phantom():
    # non-coincident transition (distinct coords, differing flags) — must NOT inject
    # a phantom mis-flagged vertex; every point keeps its own visibility flag
    pts = [(0.1, 0.1, False), (0.5, 0.5, True), (0.9, 0.9, True)]
    assert clip_polyline_unit(pts) == [(0.1, 0.1, False), (0.5, 0.5, True), (0.9, 0.9, True)]


def test_clip_preserves_coincident_transition():
    # run boundary shared as a coincident duplicate vertex (as in the real dataset):
    # both sides of the transition are represented
    pts = [(0.5, 0.9, False), (0.5, 0.6, False), (0.5, 0.6, True), (0.5, 0.3, True)]
    flags = [h for _, _, h in clip_polyline_unit(pts)]
    assert flags[0] is False and flags[-1] is True
    assert any(flags[i] != flags[i - 1] for i in range(1, len(flags)))


def test_resample_short_all_visible():
    clipped = [(0.5, 0.9, False), (0.5, 0.6, False), (0.5, 0.3, False)]
    out = resample_with_transitions(clipped, target=10, cap=12)
    assert out == [(0.5, 0.9, 1), (0.5, 0.6, 1), (0.5, 0.3, 1)]


def test_resample_preserves_transition_flags():
    clipped = [(0.5, 0.9, False), (0.5, 0.6, False), (0.5, 0.6, True), (0.5, 0.3, True)]
    out = resample_with_transitions(clipped, target=4, cap=12)
    flags = [v for _, _, v in out]
    assert 1 in flags and 0 in flags
    assert out[0][2] == 1 and out[-1][2] == 0  # starts visible, ends obstructed


def test_resample_caps_uniform_fill():
    # sparse transitions: uniform fill stays within cap
    clipped = [(round(0.1 + 0.01 * i, 3), round(0.9 - 0.01 * i, 3), False) for i in range(60)]
    out = resample_with_transitions(clipped, target=10, cap=12)
    assert len(out) <= 12
    assert all(v == 1 for _, _, v in out)


def test_resample_thins_uniform_not_transitions_under_cap():
    # one transition at index 20; dense uniform fill must be thinned to the cap,
    # but the endpoints and the transition are preserved
    clipped = [(round(0.1 + 0.02 * i, 3), 0.5, i >= 20) for i in range(30)]
    out = resample_with_transitions(clipped, target=10, cap=8)
    assert len(out) <= 8
    flags = [v for _, _, v in out]
    assert flags[0] == 1 and flags[-1] == 0
    assert 1 in flags and 0 in flags


def test_resample_preserves_all_transitions_even_beyond_cap():
    # visibility flips every point -> every point is a transition boundary,
    # so cap is soft and all labels are preserved (no silent drop)
    clipped = [(round(0.1 + 0.01 * i, 3), 0.5, bool(i % 2)) for i in range(20)]
    out = resample_with_transitions(clipped, target=10, cap=12)
    assert len(out) == 20
    flags = [v for _, _, v in out]
    assert flags == [1 if i % 2 == 0 else 0 for i in range(20)]
