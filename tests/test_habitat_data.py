from rover_vlm.habitat_data import normalize_points, clip_polyline_unit


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
