"""The cave as meshes: the ceiling, the face at the mouth, the torches along the wall."""
import numpy as np
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import landscape as ls
from openglcontext_hill_runner import scenery as s


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
        assert abs(abs(x - centre) - (ls.CAVE_HALF_WIDTH - s.TORCH_INSET)) < 1e-6
        assert np.sign(x - centre) == torch.side
        assert y - float(land.footing(x, d)) == pytest.approx(s.TORCH_HEIGHT)
        assert y < float(land.footing(x, d)) + float(land.course.ceiling(d))


# -- the ceiling ------------------------------------------------------------------------

def test_the_ceiling_lies_over_the_cave_and_nowhere_else(land, cave):
    for index in chunks_of(cave):
        origin = index * c.CHUNK_LENGTH
        ceiling = s.cave_parts(land, cave, index, origin)[s.Part.ROCK][0]
        d = origin - ceiling.positions[:, 2]
        assert d.min() >= cave.start - 1e-3 and d.max() <= cave.end + 1e-3


def test_chunks_that_hold_no_cave_have_no_cave_meshes(land, cave):
    before = c.chunk_index(cave.start) - 3
    assert s.cave_parts(land, cave, before, origin=before * c.CHUNK_LENGTH) == {}


def test_the_ceiling_is_where_the_course_says_it_is_and_faces_down(land, cave):
    index = c.chunk_index((cave.start + cave.end) / 2)
    origin = index * c.CHUNK_LENGTH
    ceiling = s.cave_parts(land, cave, index, origin)[s.Part.ROCK][0]
    d = origin - ceiling.positions[:, 2]
    x = ceiling.positions[:, 0]
    over = np.asarray(land.footing(x, d)) + np.asarray(land.course.ceiling(d))
    assert (over - ceiling.positions[:, 1]).min() >= -1e-3                  # never above it
    assert (over - ceiling.positions[:, 1]).max() <= s.CEILING_BUMP + 0.01  # and lumpy by that much
    assert (triangle_normals(ceiling)[:, 1] < 0).mean() > 0.97


def test_the_ceiling_spans_the_corridor_and_runs_into_the_walls(land, cave):
    index = c.chunk_index((cave.start + cave.end) / 2)
    ceiling = s.cave_parts(land, cave, index, index * c.CHUNK_LENGTH)[s.Part.ROCK][0]
    centre = float(land.course.centre((cave.start + cave.end) / 2))
    reach = np.abs(ceiling.positions[:, 0] - centre).max()
    assert reach >= ls.CAVE_HALF_WIDTH + 1.0


def test_the_ceiling_clears_a_runner_wherever_it_is_low(land, cave):
    for index in chunks_of(cave):
        ceiling = s.cave_parts(land, cave, index, index * c.CHUNK_LENGTH)[s.Part.ROCK][0]
        d = index * c.CHUNK_LENGTH - ceiling.positions[:, 2]
        floor = np.asarray(land.footing(ceiling.positions[:, 0], d))
        near_the_trail = np.abs(ceiling.positions[:, 0] - np.asarray(land.course.centre(d))) < 1.5
        assert (ceiling.positions[near_the_trail, 1] - floor[near_the_trail]).min() > c.STAND_HEIGHT + 0.2


# -- the mouth --------------------------------------------------------------------------

def test_the_mouth_is_a_face_above_the_opening_that_looks_back_up_the_hill(land, cave):
    index = c.chunk_index(cave.start)
    pieces = s.cave_parts(land, cave, index, index * c.CHUNK_LENGTH)[s.Part.ROCK]
    assert len(pieces) == 2
    face = pieces[1]
    assert np.allclose(triangle_normals(face)[:, 2], 1.0, atol=1e-3)
    z = index * c.CHUNK_LENGTH - cave.start
    assert np.allclose(face.positions[:, 2], z, atol=1e-3)
    centre = float(land.course.centre(cave.start))
    base = float(land.footing(centre, cave.start)) + float(land.course.ceiling(cave.start))
    assert face.positions[:, 1].min() == pytest.approx(base, abs=s.CEILING_BUMP + 0.01)
    assert face.positions[:, 1].max() >= base + ls.WALL_OVER


def test_only_the_chunk_the_cave_begins_in_has_a_face(land, cave):
    for index in chunks_of(cave):
        pieces = s.cave_parts(land, cave, index, index * c.CHUNK_LENGTH)[s.Part.ROCK]
        assert len(pieces) == (2 if index == c.chunk_index(cave.start) else 1)


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
