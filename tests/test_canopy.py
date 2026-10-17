"""The shade of the trees on the ground between them: where the cover stands in it."""
import numpy as np
import pytest

from openglcontext_hill_runner import canopy
from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import landscape as ls
from openglcontext_hill_runner import woods as w


@pytest.fixture(scope='module')
def land():
    return ls.Landscape(c.Course(3, caves=False))


@pytest.fixture(scope='module')
def shade(land):
    return canopy.Canopy(land)


def forest_chunk(land):
    """A chunk with plenty of trees."""
    return max(range(10, 40), key=lambda i: len(w.chunk_trees(land, i).heights))


def lone_tree(land):
    """A chunk's tallest tree, its place on the ground as (x, z) and its height."""
    index = forest_chunk(land)
    trees = w.chunk_trees(land, index)
    tallest = int(np.argmax(trees.heights))
    x, _y, z = (float(v) for v in trees.positions[tallest])
    return x, z, float(trees.heights[tallest])


def test_open_ground_is_in_full_sun(shade, land):
    # Sand has no trees.
    zone = next(z for z in land.zones if z.biome is ls.Biome.SAND and z.start > 300)
    d = np.linspace(zone.start + 200, zone.start + zone.length - 200, 50)
    x = np.array([land.course.centre(v) + 6.0 for v in d])
    assert np.allclose(shade(x, -d), 1.0)


def test_the_shade_is_between_the_light_that_gets_through_and_full_sun(shade, land):
    index = forest_chunk(land)
    d = index * c.CHUNK_LENGTH + np.linspace(0.0, c.CHUNK_LENGTH, 400)
    x = np.array([land.course.centre(v) + 8.0 for v in d])
    found = shade(x, -d)
    assert found.min() >= canopy.SHADOW - 1e-6 and found.max() <= 1.0
    assert (found < 1.0).any() and (found == 1.0).any()


def test_a_tree_shades_the_ground_down_the_hill_of_it_along_the_suns_way(shade, land):
    x, z, height = lone_tree(land)
    run = height * canopy.SUN[2] / -canopy.SUN[1]                  # how far down the hill its tip lands
    sideways = height * canopy.SUN[0] / -canopy.SUN[1]
    tip = (x + sideways, z + run)
    half_way = (x + sideways / 2, z + run / 2)
    assert shade(np.array([half_way[0]]), np.array([half_way[1]]))[0] < 1.0
    assert shade(np.array([tip[0]]), np.array([tip[1]]))[0] < 1.0


def test_the_shade_is_down_the_hill_of_the_trees_and_not_up_it(shade, land):
    trees = w.chunk_trees(land, forest_chunk(land))
    x, z = trees.positions[:, 0].astype(float), trees.positions[:, 2].astype(float)
    down = shade(x + 0.1, z - 5.0).mean()                    # z is -distance: smaller is down the hill
    up = shade(x + 0.1, z + 5.0).mean()
    assert down < up - 0.05


def test_a_point_is_as_shaded_either_side_of_a_chunks_edge(shade, land):
    for index in range(12, 30):
        edge = index * c.CHUNK_LENGTH
        x = np.array([land.course.centre(edge) + 7.0] * 2)
        both = shade(x, -np.array([edge - 0.01, edge + 0.01]))
        assert abs(both[0] - both[1]) < 0.75 * (1.0 - canopy.SHADOW) + 1e-6


def test_a_chunks_shade_is_made_once(land, monkeypatch):
    made = []
    original = w.chunk_trees
    monkeypatch.setattr(w, 'chunk_trees', lambda land, index: (made.append(index), original(land, index))[1])
    fresh = canopy.Canopy(land)
    x = np.array([land.course.centre(1500.0)] * 3)
    fresh(x, -np.array([1500.0, 1501.0, 1502.0]))
    once = list(made)
    fresh(x, -np.array([1500.5, 1503.0, 1504.0]))
    assert sorted(once) == [c.chunk_index(1500.0) - 1, c.chunk_index(1500.0)] and made == once


def test_it_answers_each_point_with_the_shape_of_what_it_was_asked(shade, land):
    x, d = np.zeros((3, 4)), np.full((3, 4), 1500.0)
    assert shade(x, -d).shape == (3, 4)
    assert np.shape(shade(0.0, -1500.0)) in ((), (1,))


def test_it_answers_for_points_before_the_start_and_beyond_the_land(shade):
    assert shade(np.array([0.0, 500.0, -500.0]), np.array([5.0, -100.0, -100.0])).shape == (3,)
