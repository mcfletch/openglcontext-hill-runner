"""The tile generator: chunks of the hill as streamed tiles."""
import numpy as np
import pytest
from OpenGLContext.loaders.tiles3d.generated import GeneratedTileset
from OpenGLContext.scenegraph.walk import reachable

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import landscape as ls
from openglcontext_hill_runner import scenery as s
from openglcontext_hill_runner import tiles as t
from openglcontext_hill_runner.look import make_look


@pytest.fixture(scope='module')
def look():
    return make_look(size=32)


@pytest.fixture(scope='module')
def land():
    return ls.Landscape(c.Course(3))


@pytest.fixture(scope='module')
def generator(land, look):
    return t.HillTiles(land, look)


def camera_at(distance, height=3.0):
    return np.array([0.0, height, -distance])


def shapes_of(scene):
    return [n for n in reachable(scene.group) if getattr(n, 'geometry', None) is not None]


def test_the_tiles_wanted_run_from_a_little_behind_the_camera_to_the_reach_ahead(generator):
    keys = list(generator.keys_near(camera_at(300.0), 200.0))
    assert keys == list(range(c.chunk_index(300.0 - 200.0 * t.BEHIND), c.chunk_index(500.0) + 1))


def test_no_tile_is_wanted_before_the_start_line(generator):
    assert list(generator.keys_near(camera_at(10.0), 300.0))[0] == 0


def test_a_tiles_bounds_hold_the_ground_it_covers(generator, land):
    box = generator.bounds(5)
    start = 5 * c.CHUNK_LENGTH
    for d in (start, start + c.CHUNK_LENGTH / 2, start + c.CHUNK_LENGTH):
        for x in (-s.HALF_WIDTH, 0.0, s.HALF_WIDTH):
            assert box.distance_to((x, float(land.height(x, d)), -d)) == 0.0


def test_a_generated_tile_is_a_group_of_shapes_with_a_size(generator):
    scene, nbytes = generator.generate(6)
    assert len(shapes_of(scene)) >= 5
    assert nbytes > 50_000


def test_everything_in_a_tile_lies_within_its_chunk_measured_from_the_tiles_origin(generator):
    scene, _ = generator.generate(10)
    for node in shapes_of(scene):
        z = node.geometry.positions[:, 2]
        assert z.min() >= -c.CHUNK_LENGTH - 3.0
        assert z.max() <= 3.0


def test_a_tile_is_made_around_the_start_of_its_chunk(generator):
    assert generator.origin(10) == (0.0, 0.0, -10 * c.CHUNK_LENGTH)
    box = generator.bounds(10)
    assert box.bounding_sphere()[0][2] == pytest.approx(-10.5 * c.CHUNK_LENGTH)


def test_a_tile_far_down_the_hill_keeps_its_vertices_small(generator):
    scene, _ = generator.generate(2000)
    for node in shapes_of(scene):
        assert np.abs(node.geometry.positions[:, 2]).max() < c.CHUNK_LENGTH + 30.0


def test_a_tile_placed_at_its_origin_lies_where_the_land_does(generator, land):
    tileset = GeneratedTileset(generator, reach=200.0)
    tileset.ensure(camera_at(10.5 * c.CHUNK_LENGTH))
    tile = next(t for t in tileset.root.children if t.content_key == 10)
    scene, _ = generator.generate(10)
    ground = shapes_of(scene)[0].geometry.positions.astype(float)
    world = ground + tile.world_transform[:3, 3]
    assert np.abs(world[:, 1] - land.height(world[:, 0], -world[:, 2])).max() < 0.02


def test_the_ground_and_paths_do_not_cast_shadows_and_the_obstacles_do(generator):
    casting = [n.castShadow for n in shapes_of(generator.generate(8)[0])]
    assert any(casting) and not all(casting)


def test_the_ground_of_a_tile_is_tinted_in_patches(generator):
    ground = [n.geometry for n in shapes_of(generator.generate(8)[0])
              if n.geometry.colors is not None and len(n.geometry.colors)]
    assert len(ground) >= 2
    for mesh in ground:
        assert mesh.colors.shape == (len(mesh.positions), 4)
        assert mesh.colors[:, 0].std() > 0.03


def test_where_one_country_gives_way_to_another_the_second_ground_is_laid_over_the_first(
        land, look):
    boundary = land.zones[1].start
    index = c.chunk_index(boundary)
    generator = t.HillTiles(land, look)
    scene, _ = generator.generate(index)
    meshes = [n.geometry for n in shapes_of(scene)
              if n.geometry.colors is not None and len(n.geometry.colors)]
    overlays = [m for m in meshes if m.colors[:, 3].min() < 1.0]
    assert overlays, 'a tile on a boundary has a layer over its base'
    alpha = overlays[0].colors[:, 3]
    assert alpha.min() < 0.1 and alpha.max() > 0.9


def test_the_overlay_is_blended_and_lies_a_hair_above_the_ground(land, look):
    index = c.chunk_index(land.zones[1].start)
    scene, _ = t.HillTiles(land, look).generate(index)
    overlay = next(n for n in shapes_of(scene)
                   if n.geometry.colors is not None and len(n.geometry.colors)
                   and n.geometry.colors[:, 3].min() < 1.0)
    assert overlay.appearance.alphaMode == 'BLEND'
    origin = index * c.CHUNK_LENGTH
    ground = s.terrain(land, index, origin)
    p = overlay.geometry.positions.astype(float)
    lift = p[:, 1] - land.height(p[:, 0], origin - p[:, 2])
    assert lift.min() > 0.0
    assert len(p) < len(ground.positions)


def test_a_tile_without_trees_of_its_own_holds_fewer_vertices(land, look):
    forest = next(z for z in land.zones if z.biome is ls.Biome.FOREST and z.start > 300)
    index = c.chunk_index(forest.start + forest.length / 2)
    with_trees = t.HillTiles(land, look).generate(index)[1]
    without = t.HillTiles(land, look, trees=False).generate(index)[1]
    assert without < with_trees


def test_a_chunk_where_the_paths_part_has_a_strip_for_each(generator, land):
    fork = next(f for f in land.course.forks_between(0.0, 20000.0) if f.side == 1)
    index = c.chunk_index((fork.start + fork.end) / 2)
    scene, _ = generator.generate(index)
    widest = max(float(np.ptp(n.geometry.positions[:, 0])) for n in shapes_of(scene)
                 if n.geometry.colors is not None and len(n.geometry.colors)
                 and len(n.geometry.positions) < 5000)
    assert widest > 20.0


def test_the_generator_works_with_the_engines_tileset(generator):
    tileset = GeneratedTileset(generator, reach=250.0)
    created, _ = tileset.ensure(camera_at(100.0))
    assert [tile.content_key for tile in created] == list(
        generator.keys_near(camera_at(100.0), 250.0))
