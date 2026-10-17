"""The cave as scenery: the rock of the tunnel in each chunk, and the torches along the wall."""
import numpy as np
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import landscape as ls
from openglcontext_hill_runner import scenery as s
from openglcontext_hill_runner import tunnel


@pytest.fixture(scope='module')
def land():
    return ls.Landscape(c.Course(1))


@pytest.fixture(scope='module')
def cave(land):
    return land.course.caves_between(0.0, 60000.0)[0]


def chunks_of(cave):
    return range(c.chunk_index(cave.start), c.chunk_index(cave.end) + 1)


def triangle_normals(geometry):
    p = geometry.positions[geometry.indices.reshape(-1, 3)].astype(float)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


# -- torches ----------------------------------------------------------------------------

def test_torches_stand_along_a_cave_a_few_metres_apart_on_alternate_walls(cave):
    torches = s.torches(cave)
    assert len(torches) >= 10
    assert all(cave.start < t.distance < cave.end for t in torches)
    gaps = np.diff([t.distance for t in torches])
    assert np.allclose(gaps, s.TORCH_EVERY)
    assert [t.side for t in torches] == [(-1) ** i * -1 for i in range(len(torches))]


def test_the_first_torch_is_a_little_inside_the_mouth_and_the_last_before_the_end(cave):
    torches = s.torches(cave)
    assert torches[0].distance == pytest.approx(cave.start + s.TORCH_FIRST)
    assert cave.end - torches[-1].distance >= s.TORCH_LAST_CLEAR


def test_a_torch_hangs_on_its_wall_at_head_height_inside_the_cave(land, cave):
    for torch in s.torches(cave)[:6]:
        x, y, d = s.torch_place(land, torch)
        centre = float(land.course.centre(d))
        reach = tunnel.half_width_at(s.TORCH_HEIGHT, float(land.course.ceiling(d)))
        slope = float(land.course.centre(d + 1.0) - land.course.centre(d - 1.0)) / 2.0
        assert abs(abs(x - centre) - (reach / np.hypot(1.0, slope) - s.TORCH_INSET)) < 1e-6
        assert np.sign(x - centre) == torch.side
        assert y - float(land.footing(x, d)) == pytest.approx(s.TORCH_HEIGHT)
        assert y < float(land.footing(x, d)) + float(land.course.ceiling(d))


# -- the rock ------------------------------------------------------------------------------

def test_chunks_that_hold_no_cave_have_no_cave_meshes(land, cave):
    before = c.chunk_index(cave.start) - 3
    assert s.cave_parts(land, cave, before, origin=before * c.CHUNK_LENGTH) == {}


def test_every_chunk_a_cave_runs_through_has_rock_and_the_rest_of_the_rock_is_in_none(land, cave):
    for index in chunks_of(cave):
        parts = s.cave_parts(land, cave, index, index * c.CHUNK_LENGTH)
        assert len(parts[s.Part.ROCK]) == 1 and len(parts[s.Part.ROCK][0].positions) > 1000


def test_the_torches_are_fixed_to_the_walls_where_nothing_is_cut(land, cave):
    fixings = s._fixings(land, cave)
    assert len(fixings) == len(s.torches(cave))
    bored = s.tunnel.bore(land, cave, fixings)
    for at in fixings[:5]:
        near = np.linalg.norm(bored.wall_points - at, axis=1) < 0.4
        assert near.any()
        # The wall there is the arch as it was made, within the grain: nothing was cut away.
        middle = float(land.course.centre(-at[2]))
        assert abs(abs(at[0] - middle) - ls.CAVE_HALF_WIDTH) < 0.5


# -- the torch meshes -------------------------------------------------------------------

def test_each_torch_in_a_chunk_has_a_bracket_and_a_flame(land, cave):
    for index in chunks_of(cave):
        origin = index * c.CHUNK_LENGTH
        here = [t for t in s.torches(cave) if origin <= t.distance < origin + c.CHUNK_LENGTH]
        parts = s.cave_parts(land, cave, index, origin)
        assert (s.Part.TORCH in parts) == bool(here)
        if here:
            assert len(parts[s.Part.TORCH]) == len(here) == len(parts[s.Part.FLAME])


def test_the_lights_are_at_the_flames_in_the_tiles_own_coordinates(land, cave):
    for index in chunks_of(cave):
        origin = index * c.CHUNK_LENGTH
        here = [t for t in s.torches(cave) if origin <= t.distance < origin + c.CHUNK_LENGTH]
        lights = s.cave_lights(land, cave, index, origin)
        assert len(lights) == len(here)
        for torch, at in zip(here, lights, strict=True):
            x, y, d = s.torch_place(land, torch)
            assert at == pytest.approx((x, y + s.FLAME_RISE, origin - d))
