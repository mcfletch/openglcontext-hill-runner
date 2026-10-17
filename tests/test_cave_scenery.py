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
        ceiling = s.cave_pieces(land, cave, index, origin).ceiling
        d = origin - ceiling.positions[:, 2]
        assert d.min() >= cave.start - 1e-3 and d.max() <= cave.end + 1e-3


def test_chunks_that_hold_no_cave_have_no_cave_meshes(land, cave):
    before = c.chunk_index(cave.start) - 3
    assert s.cave_parts(land, cave, before, origin=before * c.CHUNK_LENGTH) == {}
    assert all(piece is None for piece in s.cave_pieces(land, cave, before, 0.0))


def test_the_ceiling_is_where_the_course_says_it_is_less_its_lumps_and_its_arch_and_faces_down(land, cave):
    index = c.chunk_index((cave.start + cave.end) / 2)
    origin = index * c.CHUNK_LENGTH
    ceiling = s.cave_pieces(land, cave, index, origin).ceiling
    d = origin - ceiling.positions[:, 2]
    x = ceiling.positions[:, 0]
    over = np.asarray(land.footing(x, d)) + np.asarray(land.course.ceiling(d))
    hang = over - ceiling.positions[:, 1]
    assert hang.min() >= -1e-3                                        # never above it
    assert hang.max() <= s.CEILING_BUMP + s.VAULT * 2.0 + 0.01        # and no lower than lumps and arch allow
    assert (triangle_normals(ceiling)[:, 1] < 0).mean() > 0.97


def test_the_ceiling_is_arched_it_comes_down_toward_the_walls(land, cave):
    index = c.chunk_index((cave.start + cave.end) / 2)
    origin = index * c.CHUNK_LENGTH
    ceiling = s.cave_pieces(land, cave, index, origin).ceiling
    d = origin - ceiling.positions[:, 2]
    across = np.abs(ceiling.positions[:, 0] - np.asarray(land.course.centre(d)))
    hang = (np.asarray(land.footing(ceiling.positions[:, 0], d)) + np.asarray(land.course.ceiling(d))
            - ceiling.positions[:, 1])
    middle, edge = hang[across < 0.3].mean(), hang[across > ls.CAVE_HALF_WIDTH].mean()
    assert edge > middle + 0.6 * s.VAULT


def test_the_ceiling_spans_the_corridor_and_runs_into_the_walls(land, cave):
    index = c.chunk_index((cave.start + cave.end) / 2)
    ceiling = s.cave_pieces(land, cave, index, index * c.CHUNK_LENGTH).ceiling
    centre = float(land.course.centre((cave.start + cave.end) / 2))
    reach = np.abs(ceiling.positions[:, 0] - centre).max()
    assert reach >= ls.CAVE_HALF_WIDTH + 1.0


def test_the_ceiling_clears_a_runner_wherever_it_is_low(land, cave):
    for index in chunks_of(cave):
        ceiling = s.cave_pieces(land, cave, index, index * c.CHUNK_LENGTH).ceiling
        d = index * c.CHUNK_LENGTH - ceiling.positions[:, 2]
        floor = np.asarray(land.footing(ceiling.positions[:, 0], d))
        near_the_trail = np.abs(ceiling.positions[:, 0] - np.asarray(land.course.centre(d))) < 1.5
        assert (ceiling.positions[near_the_trail, 1] - floor[near_the_trail]).min() > c.STAND_HEIGHT + 0.2


# -- the top of the hill ----------------------------------------------------------------

def test_the_hill_is_closed_over_the_tunnel_by_a_strip_as_high_as_its_walls(land, cave):
    index = c.chunk_index((cave.start + cave.end) / 2)
    origin = index * c.CHUNK_LENGTH
    cap = s.cave_pieces(land, cave, index, origin).cap
    d = origin - cap.positions[:, 2]
    assert d.min() >= cave.start + s.FACE_BACK - 1e-3 and d.max() <= cave.end - s.FACE_BACK + 1e-3
    top = s._top_of_the_hill(land, d)
    assert np.abs(cap.positions[:, 1] - top).max() < 0.3
    assert (cap.normals[:, 1] > 0.7).all()


def test_the_cap_is_above_the_ceiling_by_the_rock_that_is_over_the_tunnel(land, cave):
    for index in chunks_of(cave):
        cap = s.cave_pieces(land, cave, index, index * c.CHUNK_LENGTH).cap
        if cap is None:
            continue
        d = index * c.CHUNK_LENGTH - cap.positions[:, 2]
        roof = np.asarray(land.footing(land.course.centre(d), d)) + np.asarray(land.course.ceiling(d))
        assert (cap.positions[:, 1] - roof).min() > 8.0


# -- the faces -----------------------------------------------------------------------------

def test_the_mouth_is_an_irregular_arch_in_a_face_that_slopes_back_into_the_hill(land, cave):
    index = c.chunk_index(cave.start)
    origin = index * c.CHUNK_LENGTH
    face = s.cave_pieces(land, cave, index, origin).mouth
    z_rim = origin - cave.start
    # It begins at the rim, at the mouth, and ends FACE_BACK metres down the hill.
    assert face.positions[:, 2].max() == pytest.approx(z_rim, abs=1.6)
    assert z_rim - face.positions[:, 2].min() > s.FACE_BACK - 1.6
    normals = triangle_normals(face)
    assert (normals[:, 2] > 0.1).mean() > 0.9 and (normals[:, 1] > 0.0).mean() > 0.9   # out and up
    columns = s.FACE_COLUMNS
    rim = face.positions.reshape(-1, columns, 3)[0]
    assert rim[:, 1].max() - rim[:, 1].min() > 0.8                     # an arch, not a straight edge
    assert np.ptp(face.positions[:, 2]) > s.FACE_BACK * 0.8            # not a plane


def test_the_rim_of_the_mouth_is_the_front_edge_of_the_ceiling_so_no_sky_shows_between(land, cave):
    index = c.chunk_index(cave.start)
    origin = index * c.CHUNK_LENGTH
    pieces = s.cave_pieces(land, cave, index, origin)
    rim = pieces.mouth.positions.reshape(-1, s.FACE_COLUMNS, 3)[0]
    front = pieces.ceiling.positions.reshape(-1, s.CEILING_COLUMNS, 3)[0]
    inside = np.abs(front[:, 0] - front[:, 0].mean()) < ls.CAVE_HALF_WIDTH * 0.9
    for x, y in zip(front[inside, 0], front[inside, 1], strict=True):
        assert abs(np.interp(x, rim[:, 0], rim[:, 1]) - y) < 0.15


def test_the_top_of_the_face_meets_the_top_of_the_hill(land, cave):
    index = c.chunk_index(cave.start)
    origin = index * c.CHUNK_LENGTH
    pieces = s.cave_pieces(land, cave, index, origin)
    face = pieces.mouth
    top = face.positions.reshape(-1, s.FACE_COLUMNS, 3)[-1]
    first = pieces.cap.positions[:, 2] > pieces.cap.positions[:, 2].max() - 0.01
    assert abs(float(top[:, 1].mean()) - float(pieces.cap.positions[first, 1].mean())) < 0.5


def test_the_far_end_has_a_face_too_which_looks_out_down_the_hill(land, cave):
    index = c.chunk_index(cave.end)
    origin = index * c.CHUNK_LENGTH
    face = s.cave_pieces(land, cave, index, origin).exit
    assert face is not None
    normals = triangle_normals(face)
    assert (normals[:, 2] < -0.1).mean() > 0.9 and (normals[:, 1] > 0.0).mean() > 0.9
    assert face.positions[:, 2].min() == pytest.approx(origin - cave.end, abs=1.6)


def test_a_face_is_made_in_whole_across_the_chunks_it_runs_over(land):
    land = ls.Landscape(c.Course(1))
    for cave in land.course.caves_between(0.0, 60000.0)[:12]:
        for rim, name in ((cave.start, 'mouth'), (cave.end, 'exit')):
            spans = []
            for index in range(c.chunk_index(rim) - 1, c.chunk_index(rim) + 2):
                face = getattr(s.cave_pieces(land, cave, index, index * c.CHUNK_LENGTH), name)
                if face is not None:
                    spans.append(index * c.CHUNK_LENGTH - face.positions[:, 2])
            reached = np.concatenate(spans)
            assert reached.max() - reached.min() > s.FACE_BACK - 1.6, (cave, name)


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
