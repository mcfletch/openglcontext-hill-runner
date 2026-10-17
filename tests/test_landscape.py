"""The landscape: which country the runner is in, and the ground of it."""
import numpy as np
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import landscape as ls
from openglcontext_hill_runner.landscape import Biome, Layer


@pytest.fixture(scope='module')
def land():
    return ls.Landscape(c.Course(3))


def dominant(land, x, d):
    return Biome(int(np.argmax(land.weights(np.array([x]), np.array([d]))[:, 0])))


def on_the_trail(land, d):
    return float(land.course.centre(d))


def test_the_run_starts_in_forest_and_passes_through_every_country():
    world = ls.Landscape(c.Course(3))
    seen = [dominant(world, on_the_trail(world, d), d) for d in range(0, 4000, 25)]
    assert seen[0] is Biome.FOREST
    assert set(seen) == set(Biome)


def test_a_country_goes_on_for_hundreds_of_metres():
    world = ls.Landscape(c.Course(3))
    runs, run, last = [], 0, None
    for d in range(0, 12000, 10):
        now = dominant(world, on_the_trail(world, float(d)), float(d))
        if now is last:
            run += 10
        else:
            runs.append(run)
            run, last = 10, now
    assert min(runs[1:]) >= 250


def test_the_countries_come_in_a_new_order_each_time_round():
    def names(seed):
        return [z.biome for z in ls.Landscape(c.Course(seed)).zones[:14]]

    assert names(3) == names(3)
    assert names(3) != names(4)
    assert all(a is not b for a, b in zip(names(3), names(3)[1:], strict=False))


def test_the_weights_of_the_countries_sum_to_one_and_change_smoothly(land):
    d = np.linspace(0.0, 6000.0, 6001)
    w = land.weights(np.zeros_like(d), d)
    assert w.shape == (len(Biome), len(d))
    assert np.allclose(w.sum(axis=0), 1.0)
    assert (w >= -1e-9).all()
    assert np.abs(np.diff(w, axis=1)).max() < 0.05


def test_trees_stand_thick_in_dense_forest_and_not_at_all_on_sand(land):
    def density(biome):
        zone = next(z for z in land.zones if z.biome is biome and z.start > 300)
        d = zone.start + zone.length / 2
        return float(land.tree_density(np.array([land.course.centre(d)]), np.array([d]))[0])
    assert density(Biome.SAND) < 0.02
    assert density(Biome.DENSE) > 1.5 * density(Biome.FOREST) > 0.5
    assert density(Biome.ROCKS) < 0.3 and density(Biome.MEADOW) < 0.3


def test_the_land_is_steepest_in_the_rocks_and_flattest_in_the_meadow(land):
    def bank(biome):
        zone = next(z for z in land.zones if z.biome is biome and z.start > 300)
        d = np.linspace(zone.start + 120, zone.start + zone.length - 120, 12)
        centre = np.array([land.course.centre(v) for v in d])
        ground = np.array([c.ground_height(v) for v in d])
        # The side with no fork's trail drawn away on it.
        return max(float((land.height(centre + side * 45.0, d) - ground).mean())
                   for side in (-1.0, 1.0))
    assert bank(Biome.ROCKS) > 18.0
    assert bank(Biome.MEADOW) < 8.0
    assert bank(Biome.SAND) < 9.0
    assert bank(Biome.ROCKS) > bank(Biome.DENSE) > bank(Biome.MEADOW)


def test_the_trail_is_cut_into_the_land_and_stays_level(land):
    d = np.linspace(100.0, 3000.0, 400)
    x = np.array([land.course.lane_x(1, v) for v in d])
    ground = np.array([c.ground_height(v) for v in d])
    assert np.abs(land.height(x, d) - ground).max() < 0.1


def test_the_gap_to_the_nearest_path_is_measured_from_the_paths_that_exist(land):
    d = np.array([200.0, 200.0, 200.0])
    x = np.array([land.course.lane_x(1, 200.0), land.course.lane_x(1, 200.0) + 1.0,
                  land.course.lane_x(2, 200.0) + 5.0])
    assert land.trail_gap(x, d) == pytest.approx([0.0, 0.3, 5.0])
    fork = next(f for f in land.course.forks_between(0.0, 9000.0) if f.side == 1)
    middle = (fork.start + fork.end) / 2
    branch_x = land.course.lane_x(3, middle)
    assert land.trail_gap(np.array([branch_x]), np.array([middle]))[0] == pytest.approx(0.0)
    between = (branch_x + land.course.lane_x(2, middle)) / 2
    assert land.trail_gap(np.array([between]), np.array([middle]))[0] > 10.0


def test_a_hill_stands_between_the_trail_and_a_fork_drawn_away(land):
    fork = next(f for f in land.course.forks_between(0.0, 9000.0) if f.side == 1)
    middle = (fork.start + fork.end) / 2
    a, b = land.course.lane_x(2, middle), land.course.lane_x(3, middle)
    x = np.linspace(a, b, 40)
    h = land.height(x, np.full_like(x, middle))
    assert h.max() - h[0] > 3.0 and h.max() - h[-1] > 3.0


def test_the_ground_is_uneven_under_foot_beside_the_trails_and_nearly_level_on_them(land):
    d = np.linspace(100.0, 160.0, 400)
    on = np.abs(land.relief(np.array([land.course.lane_x(1, v) for v in d]), d)).max()
    beside = np.abs(land.relief(np.array([land.course.lane_x(2, v) + 5.0 for v in d]), d)).max()
    assert on < 0.08 < beside


def test_the_ground_is_uneven_at_the_scale_of_a_metre_and_of_a_hand(land):
    d = np.full(80, 800.0)
    x = land.course.lane_x(2, 800.0) + np.arange(4.0, 24.0, 0.25)
    height = land.relief(x, d)
    assert np.std(height[::4][1:] - height[::4][:-1]) > 0.03
    assert np.std(np.diff(height)) > 0.01


def test_each_ground_layer_dominates_in_its_own_country(land):
    def layer(biome):
        zone = next(z for z in land.zones if z.biome is biome and z.start > 300)
        d = np.array([zone.start + zone.length / 2])
        return Layer(int(np.argmax(land.layer_weights(np.array([land.course.centre(d[0])]), d)[:, 0])))
    assert layer(Biome.ROCKS) is Layer.ROCK
    assert layer(Biome.SAND) is Layer.SAND
    assert layer(Biome.DENSE) is Layer.FOREST_FLOOR
    assert layer(Biome.FOREST) is Layer.GRASS and layer(Biome.MEADOW) is Layer.GRASS


def test_the_layer_weights_sum_to_one(land):
    d = np.linspace(0.0, 4000.0, 800)
    x = np.linspace(-30.0, 30.0, 800)
    assert np.allclose(land.layer_weights(x, d).sum(axis=0), 1.0)


def test_sand_is_tinted_warmer_than_forest(land):
    def tint(biome):
        zone = next(z for z in land.zones if z.biome is biome and z.start > 300)
        d = np.linspace(zone.start + 100, zone.start + zone.length - 100, 60)
        return land.tint(np.array([land.course.centre(v) + 9.0 for v in d]), d).mean(axis=0)
    sand, forest = tint(Biome.SAND), tint(Biome.FOREST)
    assert sand[0] / sand[2] > forest[0] / forest[2]
    assert land.tint(np.arange(5.0, 45.0, 0.5), np.full(80, 300.0)).shape == (80, 4)


def test_a_fork_runs_through_another_country_than_the_trail(land):
    for fork in land.course.forks_between(0.0, 9000.0)[:6]:
        middle = (fork.start + fork.end) / 2
        trunk = dominant(land, land.course.lane_x(1, middle), middle)
        branch = dominant(land, land.course.lane_x(fork.lanes[0], middle), middle)
        assert trunk is not branch, fork


def test_the_far_side_of_a_fork_is_the_countrys_own_not_the_branchs(land):
    fork = next(f for f in land.course.forks_between(0.0, 9000.0) if f.side == 1)
    before = fork.start - 150.0
    assert dominant(land, land.course.lane_x(3, before), before) is dominant(
        land, land.course.lane_x(1, before), before)


def test_the_land_has_no_step_where_a_forks_paths_begin_or_end(land):
    for fork in land.course.forks_between(0.0, 9000.0):
        for edge in (fork.start, fork.end):
            d = np.arange(edge - 4.0, edge + 4.0, 0.05)
            for lateral in (3.0, 10.0, 18.0, 30.0):
                x = np.full_like(d, land.course.centre(edge) + fork.side * lateral)
                assert np.abs(np.diff(land.height(x, d))).max() < 0.15, (edge, lateral)


def test_the_weights_still_sum_to_one_past_the_last_zone(monkeypatch):
    monkeypatch.setattr(ls, 'ZONE_LIMIT', 12)
    world = ls.Landscape(c.Course(1))
    last = world.zones[-1]
    d = np.linspace(last.start - 200.0, last.start + 3000.0, 400)
    w = world.weights(np.zeros_like(d), d)
    assert np.allclose(w.sum(axis=0), 1.0)
    assert Biome(int(np.argmax(w[:, -1]))) is last.biome


def test_a_run_of_a_day_has_countries_and_forks_all_the_way():
    world = ls.Landscape(c.Course(1))
    far = 24.0 * 3600.0 * 24.0                    # a day at the top speed
    assert world.zones[-1].start > far
    assert world.course.forks_between(far, far + 3000.0)
    d = np.array([far, far + 100.0])
    assert np.allclose(world.weights(np.zeros(2), d).sum(axis=0), 1.0)
