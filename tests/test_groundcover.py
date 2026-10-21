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
    return ls.Landscape(c.Course(3, caves=False))


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


def test_the_grass_geometry_reaches_as_far_as_it_is_asked_to_and_the_cards_no_nearer():
    art = w.forest_art()
    if art is None:
        pytest.skip("the forest demo's art is not installed")
    cover = g.Cover(art, ls.Landscape(c.Course(1)), reach=18.0)
    try:
        for node in cover.nodes:
            assert (node.clump_radius, node.card_radius) == (18.0, g.CARD_REACH)
    finally:
        cover.shutdown()


def test_the_cover_scatters_the_trail_ahead_as_the_runner_gets_on_and_dissolves_early():
    art = w.forest_art()
    if art is None:
        pytest.skip("the forest demo's art is not installed")
    land = ls.Landscape(c.Course(1))
    cover = g.Cover(art, land)
    asked = []
    try:
        for node in cover.nodes:
            assert node.fade == g.FADE
            node.prepare = lambda places, asked=asked: asked.append(places)
        cover.update(np.array([0.0, 0.0, -100.0]))
        assert len(asked) == len(cover.nodes)
        places = asked[0]
        assert [round(-z) for _x, z in places] == [130 + 30 * k for k in range(7)]
        assert places[0][0] == pytest.approx(float(land.course.centre(130.0)))
        cover.update(np.array([0.0, 0.0, -105.0]))
        assert len(asked) == len(cover.nodes)           # not again for a few metres
        cover.update(np.array([0.0, 0.0, -125.0]))
        assert len(asked) == 2 * len(cover.nodes)
    finally:
        cover.shutdown()


def test_the_cover_is_lit_by_the_shade_of_the_trees_and_a_new_hill_has_its_own():
    art = w.forest_art()
    if art is None:
        pytest.skip("the forest demo's art is not installed")
    land = ls.Landscape(c.Course(1))
    cover = g.Cover(art, land)
    try:
        first = cover.canopy
        assert cover.grass.shade is first and cover.plants.shade is first
        cover.relocate(ls.Landscape(c.Course(2)))
        assert cover.canopy is not first and cover.grass.shade is cover.canopy
    finally:
        cover.shutdown()


@needs_art
class TestScatteredInProcesses:
    """The grass and plants made by a process of their own are the ones the cover makes in line."""

    def test_the_factory_makes_the_scatter_the_cover_makes(self, land):
        art = w.forest_art()
        cover = g.Cover(art, land)
        try:
            for which, node in (('grass', cover.grass), ('plants', cover.plants)):
                maker = g.scatter_maker(land.course.seed, land.course.caves, art, g.CLUMP_REACH, which)
                rung = node.rungs[0]
                wanted = node._scatter(rung, 'cards', 20.0, -300.0, g.CARD_REACH, 2.0, rung.species.height)
                found = maker.make(('disc', 0, 'cards', 20.0, -300.0, g.CARD_REACH, 2.0, rung.species.height))
                assert len(found[0]) > 0
                assert all((a is None and b is None) or np.array_equal(a, b)
                           for a, b in zip(found, wanted, strict=True))
        finally:
            cover.shutdown()

    def test_the_factory_loads_no_meshes(self, land):
        maker = g.scatter_maker(land.course.seed, land.course.caves, w.forest_art(), g.CLUMP_REACH,
                                'grass')
        assert all(rung.clumps_far is None for rung in maker.cover.rungs)

    def test_a_cover_given_processes_grows_what_a_cover_grows_in_line(self, land):
        art = w.forest_art()
        inline, remote = g.Cover(art, land), g.Cover(art, land, processes=True)
        try:
            for node in remote.nodes:
                assert node._remote.wait_until_ready(120.0)
            for place in ((0.0, 2.0, -50.0), (0.0, 2.0, -400.0)):
                inline.update(place)
                remote.update(place)
                assert inline.wait(60.0) and remote.wait(60.0)
                inline.update(place)
                remote.update(place)
            for a, b in zip(inline.nodes, remote.nodes, strict=True):
                assert np.array_equal(a.rungs[0].cards.pos, b.rungs[0].cards.pos)
            assert all(node._remote.requests > 1 for node in remote.nodes)
        finally:
            inline.shutdown()
            remote.shutdown()

    def test_a_relocated_cover_has_processes_for_its_new_hill(self, land):
        cover = g.Cover(w.forest_art(), land, processes=True)
        try:
            first = cover.grass._remote
            cover.relocate(ls.Landscape(c.Course(4)))
            assert first.closed
            assert cover.grass._remote is not first
        finally:
            cover.shutdown()

    def test_shutting_the_cover_down_stops_its_processes(self, land):
        cover = g.Cover(w.forest_art(), land, processes=True)
        remotes = [node._remote for node in cover.nodes]
        cover.shutdown()
        assert all(remote.closed for remote in remotes)
