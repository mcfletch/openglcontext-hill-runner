"""The hill's scenery as meshes: where each piece stands, and that it is honest.

Honest means the geometry a player sees is the geometry the rules use: a log is as high as
the rules say a jump must clear, and the way under a high gate is as high as the rules say
a slide fits.
"""
import numpy as np
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import landscape as ls
from openglcontext_hill_runner import scenery as s


@pytest.fixture(scope='module')
def land():
    return ls.Landscape(c.Course(3))


def valid(geometry):
    count = len(geometry.positions)
    assert geometry.positions.shape == (count, 3)
    assert geometry.normals.shape == (count, 3)
    assert geometry.texcoords.shape == (count, 2)
    assert geometry.tangents.shape == (count, 4)
    assert np.isfinite(geometry.positions).all()
    assert geometry.indices.max() < count
    assert len(geometry.indices) % 3 == 0
    assert np.allclose(np.linalg.norm(geometry.normals, axis=1), 1.0, atol=1e-3)


def faces_agree_with_normals(geometry, fraction=0.98):
    """Triangles wind counter-clockwise seen from the side the vertex normals face."""
    triangles = geometry.indices.reshape(-1, 3)
    p = geometry.positions[triangles]
    face = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    size = np.linalg.norm(face, axis=1)
    keep = size > 1e-9
    toward = np.einsum('ij,ij->i', face[keep] / size[keep, None],
                       geometry.normals[triangles][keep].mean(axis=1))
    return (toward > 0).mean() >= fraction


def span(geometry, axis):
    values = geometry.positions[:, axis]
    return float(values.min()), float(values.max())


def extent(points, axis):
    return float(points[:, axis].min()), float(points[:, axis].max())


# -- the ground ---------------------------------------------------------------

def test_the_terrain_is_a_valid_mesh_wound_to_face_up(land):
    geometry = s.terrain(land, 2)
    valid(geometry)
    assert faces_agree_with_normals(geometry)
    assert (geometry.normals[:, 1] > 0).all()


def test_the_terrain_covers_its_chunk_from_one_edge_to_the_other(land):
    geometry = s.terrain(land, 2)
    assert span(geometry, 2) == pytest.approx((-3 * c.CHUNK_LENGTH, -2 * c.CHUNK_LENGTH))
    low, high = span(geometry, 0)
    assert low <= -s.HALF_WIDTH + 1 and high >= s.HALF_WIDTH - 1


def test_the_terrain_is_the_lands_height_at_every_vertex(land):
    geometry = s.terrain(land, 3)
    p = geometry.positions[::97].astype(float)
    assert np.allclose(p[:, 1], land.height(p[:, 0], -p[:, 2]), atol=1e-4)


def test_neighbouring_terrain_chunks_meet_without_a_seam(land):
    lower, upper = s.terrain(land, 4), s.terrain(land, 5)
    shared = -5 * c.CHUNK_LENGTH

    def row(geometry):
        at = np.isclose(geometry.positions[:, 2], shared)
        order = np.argsort(geometry.positions[at, 0])
        return geometry.positions[at][order], geometry.normals[at][order]

    (a, na), (b, nb) = row(lower), row(upper)
    assert len(a) == len(b) > 100
    assert np.allclose(a, b, atol=1e-5)
    assert np.allclose(na, nb, atol=1e-4)


def test_the_mesh_is_fine_enough_to_show_the_unevenness_of_the_ground(land):
    geometry = s.terrain(land, 2)
    rows = np.unique(np.round(geometry.positions[:, 2], 4))
    assert np.diff(rows).max() <= s.ROW_STEP + 1e-6
    near = np.unique(np.round(geometry.positions[np.abs(geometry.positions[:, 0]) < 50.0, 0], 4))
    assert np.diff(near).max() <= s.COLUMN_STEP + 1e-6


def test_each_path_is_a_strip_along_its_lane_just_above_the_land(land):
    for lane in range(c.LANE_COUNT):
        geometry = s.path(land, lane, 1)
        valid(geometry)
        assert faces_agree_with_normals(geometry)
        z = geometry.positions[:, 2]
        middle = np.array([land.course.lane_x(lane, -v) for v in z])
        assert np.abs(geometry.positions[:, 0] - middle).max() == pytest.approx(
            s.PATH_WIDTH / 2, abs=1e-3)
        ground = land.height(geometry.positions[:, 0], -z)
        lift = geometry.positions[:, 1] - ground
        assert 0.0 < lift.min() and lift.max() < 0.1


def test_a_path_that_does_not_run_through_a_chunk_has_no_strip(land):
    assert s.path(land, 4, 1) is None and s.path(land, -2, 1) is None


def test_the_paths_of_a_fork_run_the_length_of_it_and_thin_to_nothing_at_the_ends(land):
    fork = next(f for f in land.course.forks_between(0.0, 9000.0) if f.side == 1)
    lane = fork.lanes[0]
    first, last = c.chunk_index(fork.start), c.chunk_index(fork.end)
    strips = [s.path(land, lane, i) for i in range(first, last + 1)]
    assert all(strip is not None for strip in strips)
    z = np.concatenate([strip.positions[:, 2] for strip in strips])
    assert -z.min() == pytest.approx(fork.end, abs=1.0) and -z.max() == pytest.approx(
        fork.start, abs=1.0)
    widths = []
    for strip in strips:
        columns = strip.positions.reshape(-1, 3, 3)
        widths += list(columns[:, 2, 0] - columns[:, 0, 0])
    assert min(widths) < 0.05 and max(widths) == pytest.approx(s.PATH_WIDTH, abs=1e-3)


def test_the_paths_of_a_fork_are_drawn_away_from_the_trail(land):
    fork = next(f for f in land.course.forks_between(0.0, 9000.0) if f.side == 1)
    middle = (fork.start + fork.end) / 2
    strip = s.path(land, fork.lanes[0], c.chunk_index(middle))
    z = -strip.positions[:, 2]
    near = np.abs(z - middle) < 2.0
    assert strip.positions[near, 0].mean() > land.course.lane_x(2, middle) + 30.0


# -- obstacles ----------------------------------------------------------------

def parts_of(land, obstacle):
    parts = s.obstacle(land, obstacle)
    assert parts, 'an obstacle is something to look at'
    for material, geometry in parts.items():
        assert isinstance(material, s.Part)
        valid(geometry)
    return parts


def points_of(land, obstacle):
    return np.concatenate([g.positions for g in parts_of(land, obstacle).values()])


def rise(land, obstacle, points):
    x = land.course.lane_x(obstacle.lane, obstacle.distance)
    return points[:, 1] - float(land.height(x, obstacle.distance))


@pytest.mark.parametrize('kind', list(c.Kind))
def test_an_obstacle_stands_in_its_lane_and_at_its_distance(land, kind):
    obstacle = c.Obstacle(kind, 2, 100.0)
    points = points_of(land, obstacle)
    low, high = extent(points, 0)
    assert abs((low + high) / 2 - land.course.lane_x(2, 100.0)) < 0.05
    assert high - low <= 2 * c.OBSTACLE_HALF_WIDTH + 1.2
    near, far = extent(points, 2)
    assert near > -(100.0 + c.DEPTH[kind]) - 0.2 and far < -(100.0 - c.DEPTH[kind]) + 0.2


def test_a_low_obstacle_is_as_high_as_a_jump_must_clear(land):
    obstacle = c.Obstacle(c.Kind.LOW, 1, 100.0)
    assert rise(land, obstacle, points_of(land, obstacle)).max() == pytest.approx(
        c.LOW_TOP, abs=0.08)


def test_a_wall_fills_the_lane_to_the_height_the_rules_use(land):
    obstacle = c.Obstacle(c.Kind.WALL, 1, 100.0)
    points = points_of(land, obstacle)
    height = rise(land, obstacle, points)
    assert height.max() == pytest.approx(c.WALL_TOP, abs=0.1)
    assert height.min() < 0.0
    low, high = extent(points, 0)
    assert high - low >= 2 * c.OBSTACLE_HALF_WIDTH - 0.1


def test_the_way_under_a_high_gate_is_as_high_as_the_rules_say(land):
    obstacle = c.Obstacle(c.Kind.HIGH, 1, 100.0)
    points = points_of(land, obstacle)
    x = land.course.lane_x(1, 100.0)
    middle = np.abs(points[:, 0] - x) < c.OBSTACLE_HALF_WIDTH - 0.005
    assert rise(land, obstacle, points[middle]).min() == pytest.approx(c.HIGH_BOTTOM, abs=0.08)
    posts = np.abs(points[:, 0] - x) >= c.OBSTACLE_HALF_WIDTH - 0.005
    assert rise(land, obstacle, points[posts]).min() < 0.05


@pytest.mark.parametrize('kind', list(c.Kind))
def test_an_obstacle_across_two_paths_fills_both(land, kind):
    obstacle = c.Obstacle(kind, 0, 100.0, span=2)
    low, high = extent(points_of(land, obstacle), 0)
    assert low <= land.course.lane_x(0, 100.0) - c.OBSTACLE_HALF_WIDTH + 0.1
    assert high >= land.course.lane_x(1, 100.0) + c.OBSTACLE_HALF_WIDTH - 0.1


def test_a_low_obstacle_is_a_log_in_one_place_and_a_boulder_in_another(land):
    made = {tuple(sorted(p.name for p in s.obstacle(
        land, c.Obstacle(c.Kind.LOW, 1, 40.0 * i)))) for i in range(1, 40)}
    assert ('WOOD',) in made and ('ROCK',) in made


def test_a_wall_is_brick_in_one_place_and_piled_rock_in_another(land):
    made = {tuple(sorted(p.name for p in s.obstacle(
        land, c.Obstacle(c.Kind.WALL, 1, 40.0 * i)))) for i in range(1, 40)}
    assert ('BRICK',) in made and ('ROCK',) in made


def test_an_obstacle_across_a_trail_that_is_turning_turns_with_it(land):
    def heading(v):
        return (land.course.lane_x(1, v + 1) - land.course.lane_x(1, v - 1)) / 2

    def is_log(v):
        return s.Part.WOOD in s.obstacle(land, c.Obstacle(c.Kind.LOW, 1, float(v), span=2))

    d = next(v for v in np.arange(900.0, 3000.0, 3.0) if abs(heading(v)) > 0.05 and is_log(v))
    points = points_of(land, c.Obstacle(c.Kind.LOW, 1, float(d), span=2))
    # The log's length lies across the trail: z changes with x as the trail's heading says.
    assert np.polyfit(points[:, 0], points[:, 2], 1)[0] == pytest.approx(heading(d), rel=0.4)


# -- hedges, trees and rocks -------------------------------------------------------

def test_a_divider_runs_along_its_span_between_two_paths(land):
    divider = next(d for i in range(160) for d in land.course.chunk(i).dividers)
    geometry = s.divider(land, divider)
    valid(geometry)
    assert faces_agree_with_normals(geometry)
    z = geometry.positions[:, 2]
    assert (z.min(), z.max()) == pytest.approx((-divider.end, -divider.start))
    middle = np.array([(land.course.lane_x(divider.lane, -v) + land.course.lane_x(divider.lane + 1, -v)) / 2
                       for v in z])
    assert np.abs(geometry.positions[:, 0] - middle).max() <= s.HEDGE_WIDTH / 2 + 1e-3


def test_a_divider_is_as_high_as_a_hedge_that_cannot_be_hopped(land):
    divider = next(d for i in range(160) for d in land.course.chunk(i).dividers)
    geometry = s.divider(land, divider)
    p = geometry.positions
    ground = land.height(p[:, 0], -p[:, 2])
    assert (p[:, 1] - ground).max() >= s.HEDGE_HEIGHT - 0.01


def test_the_simple_trees_grow_clear_of_every_path_and_not_on_sand(land):
    forest = next(z for z in land.zones if z.biome is ls.Biome.FOREST and z.start > 300)
    sand = next(z for z in land.zones if z.biome is ls.Biome.SAND)
    chunk = land.course.chunk(c.chunk_index(forest.start + forest.length / 2))
    foliage, trunks = s.trees(land, chunk)
    for geometry in (foliage, trunks):
        valid(geometry)
    assert len(foliage.positions) > 100
    p = foliage.positions
    assert land.trail_gap(p[:, 0].astype(float), -p[:, 2].astype(float)).min() > 0.5
    empty, _ = s.trees(land, land.course.chunk(c.chunk_index(sand.start + sand.length / 2)))
    assert len(empty.positions) <= 3 * 2


def test_the_same_chunk_is_dressed_the_same_way_twice(land):
    chunk = land.course.chunk(4)
    a, b = s.trees(land, chunk), s.trees(land, chunk)
    assert np.array_equal(a[0].positions, b[0].positions)
    assert np.array_equal(s.rocks(land, chunk).positions, s.rocks(land, chunk).positions)


def test_rocks_lie_on_the_land_beside_the_trails_and_are_hand_to_boulder_sized(land):
    geometry = s.rocks(land, land.course.chunk(4))
    valid(geometry)
    assert len(geometry.positions) > 100
    p = geometry.positions.astype(float)
    assert land.trail_gap(p[:, 0], -p[:, 2]).min() >= s.ROCK_NEAREST - s.ROCK_LARGEST * 1.5
    low = p[:, 1] - land.height(p[:, 0], -p[:, 2])
    assert low.min() > -0.5 and low.max() < 2.5 * s.ROCK_LARGEST


def test_rocky_country_has_far_more_rocks_than_sand_or_meadow(land):
    def count(biome):
        zone = next(z for z in land.zones if z.biome is biome and z.start > 300)
        return len(s.rocks(land, land.course.chunk(
            c.chunk_index(zone.start + zone.length / 2))).positions)
    assert count(ls.Biome.ROCKS) > 2 * count(ls.Biome.MEADOW)
    assert count(ls.Biome.ROCKS) > 2 * count(ls.Biome.SAND)


# -- the field the ground cover is seated on -------------------------------------------

def test_the_field_answers_the_lands_height_close_enough_to_seat_a_plant(land):
    field = s.HillField(land)
    rng = np.random.default_rng(1)
    d = rng.uniform(100.0, 600.0, 400)
    x = rng.uniform(-30.0, 30.0, 400)
    assert np.abs(field.sample(x, -d) - land.height(x, d)).max() < 0.2


def test_the_field_answers_for_a_scalar_and_before_the_start_line(land):
    field = s.HillField(land)
    assert np.ndim(field.sample(0.0, -50.0)) == 0
    assert np.isfinite(field.sample(0.0, 20.0))
