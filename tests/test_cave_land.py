"""The land of a cave: rock around it, walls beside the trail, a floor cut by crevices."""
import numpy as np
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import landscape as ls
from openglcontext_hill_runner.landscape import Biome


@pytest.fixture(scope='module')
def land():
    return ls.Landscape(c.Course(1))


def caves(land, count=6):
    return land.course.caves_between(0.0, 60000.0)[:count]


def dominant(land, x, d):
    return Biome(int(np.argmax(land.weights(np.array([x]), np.array([d]))[:, 0])))


def trail(land, d):
    return float(land.course.centre(d))


def test_the_way_to_a_cave_and_the_cave_are_rock_country(land):
    for cave in caves(land):
        for d in (cave.start - 60.0, cave.start + 5.0, (cave.start + cave.end) / 2, cave.end - 5.0):
            assert dominant(land, trail(land, d), d) is Biome.ROCKS, (cave, d)


def test_the_way_out_comes_into_the_country_the_cave_leads_to(land):
    for cave in caves(land, 12):
        d = cave.end + 150.0
        expected = Biome.ROCKS if cave.exit == 'rocks' else Biome.FOREST
        assert dominant(land, trail(land, d), d) is expected, cave


def test_both_ways_out_are_taken(land):
    assert {cave.exit for cave in caves(land, 40)} == {'rocks', 'forest'}


def test_the_weights_still_sum_to_one_around_a_cave(land):
    cave = caves(land)[0]
    d = np.linspace(cave.start - 300.0, cave.end + 400.0, 700)
    assert np.allclose(land.weights(np.zeros_like(d), d).sum(axis=0), 1.0)


def test_the_floor_of_a_cave_is_the_trail(land):
    for cave in caves(land):
        d = (cave.start + cave.end) / 2
        assert float(land.height(trail(land, d), d)) == pytest.approx(float(c.ground_height(d)), abs=0.35)


def test_the_valley_over_a_cave_is_filled_to_the_height_of_the_hills_beside_it(land):
    for cave in caves(land):
        d = (cave.start + cave.end) / 2
        centre = trail(land, d)
        top = float(land.surface(centre, d))
        floor = float(land.height(centre, d))
        assert top - floor > float(land.course.ceiling(d)) + 8.0, cave
        beside = [float(land.surface(centre + side * 40.0, d)) for side in (-1.0, 1.0)]
        assert abs(top - np.mean(beside)) < 0.5 * (top - floor), cave


def test_there_is_no_trench_along_the_top_of_a_cave(land):
    cave = caves(land)[0]
    d = np.linspace(cave.start + ls.FACE_LENGTH + 5.0, cave.end - ls.FACE_LENGTH - 5.0, 20)
    centre = np.array([trail(land, v) for v in d])
    across = [np.asarray(land.surface(centre + dx, d)) - np.asarray(land.surface(centre, d))
              for dx in (-6.0, -3.0, 3.0, 6.0)]
    assert np.abs(across).max() < 4.0


def test_the_surface_is_the_height_wherever_there_is_no_cave(land):
    cave = caves(land)[0]
    for d in (cave.start - 300.0, cave.start - 1.0, cave.end + 1.0, cave.end + 300.0):
        x = trail(land, d) + np.array([-30.0, -3.0, 0.0, 4.0, 25.0])
        assert np.allclose(land.surface(x, np.full(5, d)), land.height(x, np.full(5, d), crevices=False))


def test_the_hill_rises_from_the_floor_at_the_mouth_and_falls_to_it_at_the_far_end(land):
    cave = caves(land)[0]
    for start, sign in ((cave.start, 1.0), (cave.end, -1.0)):
        d = start + sign * np.linspace(0.0, ls.FACE_LENGTH, 12)
        x = np.array([trail(land, v) for v in d])
        rise = np.asarray(land.surface(x, d)) - np.asarray(land.footing(x, d))
        assert rise[0] == pytest.approx(0.0, abs=0.05)
        assert rise[-1] > 14.0
        assert np.all(np.diff(rise) > 0.0)


def test_no_trees_grow_over_a_cave_or_stand_on_the_hill_above_it(land):
    cave = caves(land)[0]
    d = (cave.start + cave.end) / 2
    for dx in (0.0, 8.0, 30.0, 70.0):
        assert float(land.tree_density(trail(land, d) + dx, d)) == pytest.approx(0.0, abs=1e-9)
        assert float(land.cover_density(trail(land, d) + dx, d)) == pytest.approx(0.0, abs=1e-9)
        assert float(land.grass_density(trail(land, d) + dx, d)) == pytest.approx(0.0, abs=1e-9)


def test_trees_grow_as_they_did_well_away_from_a_cave(land):
    cave = caves(land)[0]
    d = cave.end + 4 * ls.FACE_LENGTH + 400.0
    assert float(land.tree_density(trail(land, d) + 20.0, d)) >= 0.0


# -- crevices ---------------------------------------------------------------------------

def a_crevice(land):
    for cave in caves(land, 12):
        for o in land.course.obstacles_between(cave.start, cave.end):
            if o.kind is c.Kind.CREVICE:
                return o
    pytest.fail('no crevice in the first caves')


def test_a_crevice_is_cut_across_the_floor(land):
    crevice = a_crevice(land)
    d = crevice.distance
    x = trail(land, d)
    assert float(land.footing(x, d)) - float(land.height(x, d)) == pytest.approx(ls.CREVICE_DEPTH, abs=0.05)
    for lane in range(c.LANE_COUNT):
        xl = land.course.lane_x(lane, d)
        assert float(land.footing(xl, d)) - float(land.height(xl, d)) > ls.CREVICE_DEPTH - 0.2


def test_a_crevice_is_as_wide_as_the_obstacle_and_no_wider(land):
    crevice = a_crevice(land)
    x = trail(land, crevice.distance)
    d = np.linspace(crevice.distance - 3.0, crevice.distance + 3.0, 601)
    cut = np.asarray(land.height(x + 0 * d, d)) - np.asarray(land.footing(x + 0 * d, d))
    inside = np.abs(cut) > 0.5
    width = d[inside].max() - d[inside].min()
    assert c.DEPTH[c.Kind.CREVICE] * 0.55 < width < c.DEPTH[c.Kind.CREVICE]


def test_the_runners_footing_over_a_crevice_runs_on_from_the_lip_either_side(land):
    crevice = a_crevice(land)
    x, d = trail(land, crevice.distance), crevice.distance
    before, over, after = (float(land.footing(x, d + o)) for o in (-1.5, 0.0, 1.5))
    assert before > over > after and (before - over) == pytest.approx(over - after, abs=0.3)


def test_the_footing_is_the_height_where_there_is_no_crevice(land):
    cave = caves(land)[0]
    d = np.linspace(cave.start, cave.end, 50)
    x = np.array([trail(land, v) for v in d]) + 3.0
    crevices = [o.distance for o in land.course.obstacles_between(cave.start, cave.end)
                if o.kind is c.Kind.CREVICE]
    away = np.array([all(abs(v - m) > 2.0 for m in crevices) for v in d])
    assert np.allclose(land.footing(x, d)[away], land.height(x, d)[away])
