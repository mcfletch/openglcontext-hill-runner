"""The ground cover: grass to the edge of the trails, taller than the forest demo grows it, and
the plants that grow with it kept back from them."""
import numpy as np
import pytest
from OpenGLContext.scenegraph.vegetation.cover import CARD_SUN, CoverSpecies

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import groundcover as g
from openglcontext_hill_runner import landscape as ls
from openglcontext_hill_runner import woods as w


@pytest.fixture(scope='module')
def land():
    return ls.Landscape(c.Course(3))


def species(name, height=0.2):
    return CoverSpecies(name=name, card=name + '.png', density=2.0, height=height)


def where(land, biome, beside=3.0):
    zone = next(z for z in land.zones if z.biome is biome and z.start > 300)
    d = zone.start + zone.length / 2
    return land.course.lane_x(1, d) + beside, d


def test_the_species_split_into_grasses_and_the_plants_that_grow_with_them():
    grasses, plants = g.split([species('grass_medium_01'), species('fern_02'),
                               species('grass_medium_02'), species('shrub_04')])
    assert [s.name for s in grasses] == ['grass_medium_01', 'grass_medium_02']
    assert [s.name for s in plants] == ['fern_02', 'shrub_04']


def test_the_grass_stands_taller_than_it_was_drawn_and_the_rest_is_left_as_it_was():
    grasses, plants = g.split([species('grass_a', 0.15), species('fern', 0.4)])
    assert grasses[0].height == pytest.approx(0.15 * g.TALL)
    assert plants[0].height == pytest.approx(0.4)
    assert grasses[0].name == 'grass_a' and grasses[0].density == 2.0


def test_every_card_is_lit_as_much_as_the_geometry_it_stands_in_for_whatever_the_pack_records():
    dim = [species('grass_a').varied(sunLevel=0.42), species('fern').varied(sunLevel=0.42)]
    grasses, plants = g.split(dim)
    assert [s.sunLevel for s in (*grasses, *plants)] == [CARD_SUN, CARD_SUN]


def test_meadow_is_mostly_grass_and_forest_has_more_of_the_plants_among_the_trees(land):
    meadow, forest = where(land, ls.Biome.MEADOW), where(land, ls.Biome.FOREST)
    grass, plants = g.grass_where(land), g.plants_where(land)
    at = lambda f, p: float(f(np.array([p[0]]), np.array([-p[1]]))[0])  # noqa: E731 reads best inline
    assert at(grass, meadow) > 0.9 > at(plants, meadow)
    assert at(plants, forest) > at(plants, meadow) * 3


def test_nothing_grows_on_a_trail_and_grass_comes_nearer_to_it_than_the_other_plants(land):
    _, d = where(land, ls.Biome.FOREST)
    grass, plants = g.grass_where(land), g.plants_where(land)
    z = np.array([-d])
    trail = np.array([land.course.lane_x(1, d)])
    assert grass(trail, z)[0] == 0.0 and plants(trail, z)[0] == 0.0
    beside = np.array([land.course.lane_x(2, d) + 0.45])
    assert land.trail_gap(beside, np.array([d]))[0] == pytest.approx(0.45, abs=0.02)
    assert grass(beside, z)[0] > 0.3 and plants(beside, z)[0] == 0.0


def test_both_weights_are_proportions(land):
    x = np.linspace(-30, 30, 200)
    for biome in ls.Biome:
        d = next(z.start for z in land.zones if z.biome is biome and z.start > 300) + 50
        for f in (g.grass_where(land), g.plants_where(land)):
            weight = f(x, np.full(200, -d))
            assert weight.min() >= 0.0 and weight.max() <= 1.0


needs_art = pytest.mark.skipif(w.forest_art() is None, reason="the forest demo's art is not installed")


@needs_art
def test_the_cover_is_two_nodes_the_grass_and_the_rest(land):
    cover = g.Cover(w.forest_art(), land)
    try:
        assert len(cover.nodes) == 2
        names = [[r.species.name for r in node.rungs] for node in cover.nodes]
        assert all(n.startswith('grass') for n in names[0])
        assert not any(n.startswith('grass') for n in names[1]) and names[1]
    finally:
        cover.shutdown()


@needs_art
def test_relocated_cover_keeps_its_nodes(land):
    cover = g.Cover(w.forest_art(), land)
    try:
        nodes = list(cover.nodes)
        cover.relocate(ls.Landscape(c.Course(4)))
        assert cover.nodes == nodes
    finally:
        cover.shutdown()


@needs_art
def test_the_cover_can_be_waited_for(land):
    cover = g.Cover(w.forest_art(), land)
    try:
        cover.update((0.0, 2.0, -50.0))
        assert cover.wait(60.0)
    finally:
        cover.shutdown()


def test_the_cover_reaches_as_far_as_its_cards_and_has_no_sparse_field_beyond():
    art = w.forest_art()
    if art is None:
        pytest.skip("the forest demo's art is not installed")
    cover = g.Cover(art, ls.Landscape(c.Course(1)))
    try:
        for node in cover.nodes:
            assert (node.clump_radius, node.card_radius) == (g.CLUMP_REACH, g.CARD_REACH)
            assert not node.has_far
    finally:
        cover.shutdown()
