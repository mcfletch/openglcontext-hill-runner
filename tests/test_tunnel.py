"""The tunnel: an arch swept along the trail, cut into rock, and where it comes out of the hill."""
import numpy as np
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import landscape as ls
from openglcontext_hill_runner import portal as pt
from openglcontext_hill_runner import scenery as s
from openglcontext_hill_runner import tunnel as t


@pytest.fixture(scope='module')
def land():
    return ls.Landscape(c.Course(1))


@pytest.fixture(scope='module')
def cave(land):
    return land.course.caves_between(0.0, 60000.0)[0]


@pytest.fixture(scope='module')
def bored(land, cave):
    return t.bore(land, cave, s._fixings(land, cave))


# -- the section ---------------------------------------------------------------------------

def test_the_wall_is_upright_to_where_the_arch_begins_and_comes_in_above_it():
    height = 6.0
    top = height * (1.0 - t.ARCH_DROP)
    assert t.half_width_at(0.0, height) == ls.CAVE_HALF_WIDTH
    assert t.half_width_at(top - 0.01, height) == ls.CAVE_HALF_WIDTH
    assert t.half_width_at(top + 0.5, height) < ls.CAVE_HALF_WIDTH
    assert t.half_width_at(height - 0.01, height) < 0.7 * ls.CAVE_HALF_WIDTH


def test_the_sections_stand_close_over_the_faces_at_either_end_and_include_both_ends(cave):
    d = t._stations(cave)
    assert cave.start in d and cave.end in d
    assert np.all(np.diff(d) > 1e-3)
    near = d[(d > cave.start) & (d < cave.start + ls.FACE_LENGTH)]
    middle = d[(d > cave.start + 60.0) & (d < cave.end - 60.0)]
    assert np.diff(near).max() <= t.NEAR_STEP + 1e-6
    assert np.diff(middle).min() >= t.FAR_STEP - 1e-6 and d[0] < cave.start < d[-1]


# -- where it comes out of the hill ------------------------------------------------------------

@pytest.mark.parametrize('which', ['mouth', 'exit'])
def test_the_rim_of_each_opening_lies_on_the_surface_of_the_hill(land, bored, which):
    rim = getattr(bored, which)
    d = -rim[:, 2]
    assert np.abs(rim[:, 1] - np.asarray(land.surface(rim[:, 0], d))).max() < 0.08


def test_the_rims_run_from_the_foot_of_one_wall_over_the_arch_to_the_foot_of_the_other(land, bored, cave):
    for rim, at in ((bored.mouth, cave.start), (bored.exit, cave.end)):
        feet = rim[[0, -1]]
        assert np.abs(-feet[:, 2] - at).max() < 1.0
        assert np.abs(feet[:, 1] - np.asarray(land.footing(feet[:, 0], -feet[:, 2]))).max() < 0.1
        assert rim[:, 1].max() - feet[:, 1].min() > 2.0


def test_the_floor_begins_and_ends_at_the_feet_of_the_walls(bored):
    for rim, edge in ((bored.mouth, bored.mouth_floor), (bored.exit, bored.exit_floor)):
        assert np.allclose(edge[0], rim[0], atol=1e-6) and np.allclose(edge[-1], rim[-1], atol=1e-6)
        assert len(edge) == t.FLOOR_COLUMNS


# -- the walls --------------------------------------------------------------------------------

def test_the_walls_face_into_the_tunnel(land, bored):
    p = bored.wall_points[bored.wall_triangles]
    face = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-12)
    d = -p[:, :, 2].mean(axis=1)
    middle = np.stack([np.asarray(land.course.centre(d)),
                       np.asarray(land.footing(land.course.centre(d), d)) + 1.5, -d], axis=1)
    toward = np.sum(face * (middle - p.mean(axis=1)), axis=1)
    assert (toward > 0.0).mean() > 0.9


def test_the_tunnel_is_never_lower_than_the_courses_ceiling_over_the_lanes(land, bored, cave):
    rng = np.random.default_rng(3)
    for d in rng.uniform(cave.start + 30.0, cave.end - 30.0, 25):
        for lane in range(c.LANE_COUNT):
            x = land.course.lane_x(lane, d)
            near = (np.abs(bored.wall_points[:, 0] - x) < 0.35) & (np.abs(-bored.wall_points[:, 2] - d) < 0.55) \
                & (bored.wall_points[:, 1] > float(land.footing(x, d)) + 1.0)
            assert near.any()
            clearance = bored.wall_points[near, 1].min() - float(land.footing(x, d))
            assert clearance > float(land.course.ceiling(d)) * (1.0 - t.ARCH_DROP * 0.2) - 0.1


def test_the_rock_is_cut_in_facets_and_pits_and_not_a_smooth_tube(land, bored, cave):
    d = -bored.wall_points[:, 2]
    inside = (d > cave.start + 60.0) & (d < cave.end - 60.0) & (bored.wall_points[:, 1]
                                                                > np.asarray(land.footing(bored.wall_points[:, 0], d)) + 0.5)
    reach = np.abs(bored.wall_points[inside, 0] - np.asarray(land.course.centre(d[inside])))
    top = bored.wall_points[inside, 1] - np.asarray(land.footing(bored.wall_points[inside, 0], d[inside]))
    smooth = np.array([t.half_width_at(float(y), float(land.course.ceiling(float(v))))
                       for y, v in zip(top[::40], d[inside][::40], strict=True)])
    deviation = reach[::40] - smooth
    assert deviation.max() > 0.3
    assert deviation.min() > -0.5


def test_the_walls_are_closed_except_at_the_rims_and_the_feet(land, bored):
    triangles = bored.wall_triangles
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    keys, counts = np.unique(edges, axis=0, return_counts=True)
    assert counts.max() <= 2
    loose = np.unique(keys[counts == 1])
    p = bored.wall_points[loose]
    rims = np.concatenate([bored.mouth, bored.exit])
    on_rim = np.array([np.abs(rims - v).sum(axis=1).min() < 1e-6 for v in p])
    at_foot = p[:, 1] - np.asarray(land.footing(p[:, 0], -p[:, 2])) < 1e-3
    assert (on_rim | at_foot).all()


# -- the floor ----------------------------------------------------------------------------------

def test_the_floor_is_the_height_of_the_trail_and_is_cut_by_the_crevices(land, bored, cave):
    p = bored.floor_points.reshape(-1, t.FLOOR_COLUMNS, 3)
    middle = p[:, 4:-4].reshape(-1, 3)
    d = -middle[:, 2]
    assert np.abs(middle[:, 1] - np.asarray(land.height(middle[:, 0], d))).max() < 0.1
    flat = bored.floor_points
    assert (np.asarray(land.footing(flat[:, 0], -flat[:, 2])) - flat[:, 1]).max() > ls.CREVICE_DEPTH - 0.4
    corners = bored.floor_points[bored.floor_triangles]
    assert np.median(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 1]) > 0.0


# -- in chunks ------------------------------------------------------------------------------------

def test_every_triangle_of_the_tunnel_is_in_exactly_one_chunk(land, bored, cave):
    total = 0
    for index in range(c.chunk_index(cave.start) - 1, c.chunk_index(cave.end) + 2):
        piece = bored.piece(index, index * c.CHUNK_LENGTH)
        if piece is not None:
            total += len(piece.positions) // 3
            assert piece.positions[:, 2].min() > -c.CHUNK_LENGTH - 8.0
            assert piece.positions[:, 2].max() < 8.0
    assert total == len(bored.wall_triangles) + len(bored.floor_triangles)


# -- the ground about the openings ------------------------------------------------------------------

@pytest.mark.parametrize('which', ['mouth', 'exit'])
def test_the_window_is_on_the_lattice_and_holds_the_opening(bored, which):
    patch = getattr(bored, which + '_ground')
    rim = getattr(bored, which)
    x0, x1, d0, d1 = patch.window
    for v in (x0, x1, d0, d1):
        assert v / pt.LATTICE == pytest.approx(round(v / pt.LATTICE))
    assert x0 < rim[:, 0].min() - 1.0 and x1 > rim[:, 0].max() + 1.0
    assert d0 < (-rim[:, 2]).min() - 1.0 and d1 > (-rim[:, 2]).max() + 1.0


@pytest.mark.parametrize('which', ['mouth', 'exit'])
def test_the_patch_fills_the_window_less_the_opening_and_meets_the_rim_exactly(bored, which):
    patch = getattr(bored, which + '_ground')
    rim = getattr(bored, which)
    x0, x1, d0, d1 = patch.window
    corners = patch.points[patch.triangles]
    plan = corners[:, :, [0, 2]]
    area = 0.5 * np.abs((plan[:, 1, 0] - plan[:, 0, 0]) * (plan[:, 2, 1] - plan[:, 0, 1])
                        - (plan[:, 2, 0] - plan[:, 0, 0]) * (plan[:, 1, 1] - plan[:, 0, 1])).sum()
    floor = getattr(bored, which + '_floor')
    loop = np.concatenate([rim, floor[::-1][1:-1]])
    x, d = loop[:, 0], -loop[:, 2]
    hole = 0.5 * abs(float(np.sum(x * np.roll(d, -1) - np.roll(x, -1) * d)))
    assert area == pytest.approx((x1 - x0) * (d1 - d0) - hole, rel=0.01)
    assert all(np.abs(patch.points - v).sum(axis=1).min() < 1e-6 for v in loop)


def test_the_ground_has_no_cell_inside_the_window_and_the_patch_takes_their_place(land, cave):
    index = c.chunk_index(cave.start)
    origin = index * c.CHUNK_LENGTH
    ground = s.terrain(land, index, origin)
    patch = s._portals(land, origin, origin + c.CHUNK_LENGTH)[0]
    x0, x1, d0, d1 = patch.window
    triangles = ground.positions[ground.indices.reshape(-1, 3)].astype(float)
    middle = triangles.mean(axis=1)
    d = origin - middle[:, 2]
    lattice = triangles[:, :, 0].std(axis=1) < 0.6
    inside = (middle[:, 0] > x0 + 0.01) & (middle[:, 0] < x1 - 0.01) & (d > d0 + 0.01) & (d < d1 - 0.01)
    assert not (inside & lattice & (np.abs(d - d0) > 0.0)).sum() > len(patch.triangles)
    assert len(ground.positions) > 70_000 // 2
