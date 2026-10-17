"""The woods: the forest demo's trees, planted along the hill a window at a time."""
import os

import numpy as np
import pytest
from OpenGLContext.scenegraph.vegetation.field import VegetationField

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import landscape as ls
from openglcontext_hill_runner import woods as w


@pytest.fixture(scope='module')
def land():
    return ls.Landscape(c.Course(3))


def in_country(land, biome):
    """A chunk in the middle of a stretch of ``biome`` that no fork runs through."""
    for zone in land.zones:
        if zone.biome is not biome or zone.start < 300:
            continue
        for index in range(c.chunk_index(zone.start + 100), c.chunk_index(zone.start + zone.length - 100)):
            lo, hi = index * c.CHUNK_LENGTH - 40, (index + 1) * c.CHUNK_LENGTH + 40
            if not land.course.forks_between(lo, hi):
                return index
    raise AssertionError('no clear stretch of %s' % biome)


def test_a_chunks_trees_are_the_same_every_time(land):
    a, b = w.chunk_trees(land, 7), w.chunk_trees(land, 7)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.kinds, b.kinds)


def test_trees_lie_in_the_chunk_they_are_made_for(land):
    for index in (in_country(land, ls.Biome.FOREST), in_country(land, ls.Biome.DENSE)):
        trees = w.chunk_trees(land, index)
        assert len(trees.positions) > 20
        d = -trees.positions[:, 2]
        assert d.min() >= index * c.CHUNK_LENGTH and d.max() < (index + 1) * c.CHUNK_LENGTH


def test_the_trails_run_through_a_clearing_no_crown_overhangs(land):
    for index in (in_country(land, ls.Biome.FOREST), 12, 60):
        trees = w.chunk_trees(land, index)
        crowns = np.array([w.KINDS[k][-3] for k in trees.kinds]) * trees.heights / 2.0
        gap = land.trail_gap(trees.positions[:, 0].astype(float),
                             -trees.positions[:, 2].astype(float))
        assert (gap - crowns >= w.CLEARING).all()


def test_trees_stand_clear_of_the_paths_of_a_fork_as_well(land):
    fork = next(f for f in land.course.forks_between(0.0, 9000.0) if f.side == 1)
    index = c.chunk_index((fork.start + fork.end) / 2)
    trees = w.chunk_trees(land, index)
    x, d = trees.positions[:, 0].astype(float), -trees.positions[:, 2].astype(float)
    assert len(x) > 20
    assert land.trail_gap(x, d).min() >= w.CLEARING


def test_trees_stand_on_the_land_with_their_roots_sunk_in_it(land):
    trees = w.chunk_trees(land, in_country(land, ls.Biome.FOREST))
    ground = land.height(trees.positions[:, 0].astype(float), -trees.positions[:, 2].astype(float))
    sunk = ground - trees.positions[:, 1]
    assert (sunk >= -1e-3).all() and (sunk <= 0.2 * trees.heights).all()


def test_no_two_trees_stand_in_each_others_crowns(land):
    trees = w.chunk_trees(land, in_country(land, ls.Biome.FOREST))
    p = trees.positions[:, [0, 2]]
    gaps = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(len(p)) * 1e9
    assert gaps.min() >= w.SPACING * 0.7


def test_conifers_and_maples_both_grow(land):
    index = in_country(land, ls.Biome.FOREST)
    kinds = np.concatenate([w.chunk_trees(land, i).kinds for i in range(index, index + 3)])
    assert set(np.unique(kinds)) & set(w.CONIFERS) and set(np.unique(kinds)) & set(w.MAPLES)
    assert kinds.max() < len(w.KINDS)


def test_heights_are_a_forests(land):
    heights = w.chunk_trees(land, in_country(land, ls.Biome.FOREST)).heights
    assert 8.0 <= heights.min() and heights.max() <= 22.0


def test_dense_forest_has_more_trees_than_forest_and_sand_has_none(land):
    def count(biome):
        return len(w.chunk_trees(land, in_country(land, biome)).positions)

    assert count(ls.Biome.DENSE) > 1.6 * count(ls.Biome.FOREST) > 20
    assert count(ls.Biome.SAND) == 0
    assert count(ls.Biome.MEADOW) < 0.4 * count(ls.Biome.FOREST)
    assert count(ls.Biome.ROCKS) < 0.4 * count(ls.Biome.FOREST)


def test_the_window_around_a_distance_holds_what_is_ahead_and_a_little_behind():
    first, count = w.window_for(1000.0)
    assert first * c.CHUNK_LENGTH <= 1000.0 - w.BEHIND
    assert (first + count) * c.CHUNK_LENGTH >= 1000.0 + w.AHEAD


def test_a_window_never_starts_before_the_hill():
    assert w.window_for(10.0)[0] == 0


art = w.forest_art()
needs_art = pytest.mark.skipif(art is None, reason="the forest demo's art is not installed")


def test_without_the_forest_demos_art_there_is_none(monkeypatch):
    monkeypatch.setattr(w, '_art_directory', lambda: (_ for _ in ()).throw(ImportError('no')))
    assert w.forest_art() is None


@needs_art
def test_the_species_name_files_that_exist():
    for species in w.species(art):
        for path in (species.mesh, species.solidTexture, species.foliageTexture,
                     species.impostor):
            assert os.path.isfile(path), path
    assert len(w.species(art)) == len(w.KINDS)


@needs_art
def test_the_woods_plant_a_field_of_the_window_ahead(land):
    woods = w.Woods(w.species(art), land)
    woods.update((0.0, 2.0, -500.0))
    assert isinstance(woods.field, VegetationField)
    first, count = woods.window
    assert first * c.CHUNK_LENGTH <= 500.0 - w.BEHIND
    assert woods.field.tree_count > 500


@needs_art
def test_a_runner_still_in_the_window_keeps_its_field(land):
    woods = w.Woods(w.species(art), land)
    woods.update((0.0, 2.0, -500.0))
    kept = woods.field
    woods.update((0.0, 2.0, -560.0))
    assert woods.field is kept


@needs_art
def test_a_runner_near_the_end_of_the_window_gets_the_next_one_in_the_same_field(land):
    woods = w.Woods(w.species(art), land)
    woods.update((0.0, 2.0, -500.0))
    field, (first, count) = woods.field, woods.window
    near_the_end = (first + count) * c.CHUNK_LENGTH - w.AHEAD / 4 + 1.0
    woods.update((0.0, 2.0, -near_the_end))
    assert woods.window == (first, count)
    woods.wait()
    woods.update((0.0, 2.0, -near_the_end))
    assert woods.window[0] > first
    assert woods.field is field
    assert woods.children == [field]


@needs_art
def test_the_woods_of_another_landscape_replace_these_in_the_same_field(land):
    woods = w.Woods(w.species(art), land)
    woods.update((0.0, 2.0, -500.0))
    field, before = woods.field, woods.field.positions.copy()
    woods.relocate(ls.Landscape(c.Course(4)))
    assert woods.field is field and woods.window[0] == 0
    assert field.positions.shape != before.shape or not np.allclose(field.positions, before)


@needs_art
def test_a_window_with_no_trees_in_it_still_plants_a_field(land, monkeypatch):
    monkeypatch.setattr(w, 'chunk_trees', lambda land, index: w.Trees(
        np.zeros((0, 3), 'f'), np.zeros(0, 'f'), np.zeros(0, 'f'), np.zeros(0, int)))
    woods = w.Woods(w.species(art), land)
    woods.update((0.0, 2.0, -500.0))
    assert woods.field is not None and woods.field.tree_count == 1
