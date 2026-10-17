"""The air of the hill: bright in the open, dark and close in a cave."""
import numpy as np
import pytest

from openglcontext_hill_runner import atmosphere as a
from openglcontext_hill_runner import course as c


@pytest.fixture(scope='module')
def world():
    return c.Course(1)


@pytest.fixture(scope='module')
def cave(world):
    return world.caves_between(0.0, 20000.0)[0]


def test_it_is_as_bright_as_open_air_outside_a_cave(world, cave):
    assert a.darkness(world, cave.start - 5.0) == 0.0
    assert a.darkness(world, cave.end + 5.0) == 0.0
    assert a.darkness(world, 500.0) == 0.0


def test_it_is_darkest_deep_in_a_cave(world, cave):
    assert a.darkness(world, (cave.start + cave.end) / 2) == pytest.approx(1.0)


def test_the_light_goes_over_a_few_metres_inside_each_end_not_at_the_edge_of_the_rock(world, cave):
    inside = [a.darkness(world, cave.start + t * a.LIGHT_FADE) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert inside[0] == 0.0 and inside[-1] == pytest.approx(1.0)
    assert inside == sorted(inside)
    outside = [a.darkness(world, cave.end - t * a.LIGHT_FADE) for t in (0.0, 0.5, 1.0)]
    assert outside[0] == pytest.approx(0.0, abs=1e-6) and outside[-1] == pytest.approx(1.0)


def test_the_open_air_is_the_games_haze_and_the_environment_at_full(world):
    air = a.air_at(world, 500.0, haze=(0.8, 0.87, 0.92), range_=260.0)
    assert (air.ibl, air.sun) == (1.0, 1.0)
    assert air.fog_colour == pytest.approx((0.8, 0.87, 0.92))
    assert air.fog_range == pytest.approx(260.0)


def test_deep_in_a_cave_the_environment_light_is_nearly_out_and_the_fog_is_dark(world, cave):
    air = a.air_at(world, (cave.start + cave.end) / 2, haze=(0.8, 0.87, 0.92), range_=260.0)
    assert air.ibl < 0.1 and air.sun < 0.5
    assert max(air.fog_colour) < 0.1
    assert air.fog_range == pytest.approx(a.CAVE_RANGE)


def test_the_air_changes_smoothly_through_a_mouth(world, cave):
    d = np.linspace(cave.start - 5.0, cave.start + 2 * a.LIGHT_FADE, 200)
    ibl = np.array([a.air_at(world, v, haze=(0.8, 0.87, 0.92), range_=260.0).ibl for v in d])
    assert np.abs(np.diff(ibl)).max() < 0.05
