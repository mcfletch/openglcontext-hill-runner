"""The materials: procedural ones always, photographic ones where they can be had."""
import numpy as np
import pytest
from OpenGLContext.loaders import cc0
from OpenGLContext.scenegraph.physicalmaterial import PhysicalMaterial

from openglcontext_hill_runner import look
from openglcontext_hill_runner.landscape import Layer
from openglcontext_hill_runner.scenery import Part


def test_every_part_of_the_scenery_has_a_material():
    made = look.make_look(size=32)
    assert set(made.parts) == set(Part)


def test_without_a_network_the_ground_is_procedural(monkeypatch):
    monkeypatch.setattr(cc0, 'physical_material', lambda name, **named: None)
    made = look.make_look(size=32, photos=True)
    assert set(made.ground) == set(Layer) and made.path is not None
    assert all(m.baseTexture for m in made.ground.values())      # procedural maps, still textured


def test_photographs_are_used_for_the_ground_the_paths_and_the_rocks_and_bark_when_asked_for(
        monkeypatch):
    asked = []

    def photograph(name, **named):
        asked.append((name, named))
        return PhysicalMaterial()

    monkeypatch.setattr(cc0, 'physical_material', photograph)
    made = look.make_look(size=32, photos=True)
    names = [name for name, _ in asked]
    assert {'grass', 'forest_floor', 'rock', 'sand', 'dirt', 'bark'} <= set(names)
    assert ('bark', {'tint': look.WEATHERED}) in asked
    assert isinstance(made.parts[Part.BARK], PhysicalMaterial)


def test_without_photos_nothing_is_asked_for(monkeypatch):
    monkeypatch.setattr(cc0, 'physical_material', lambda name, **named: pytest.fail(name))
    made = look.make_look(size=32, photos=False)
    assert set(made.ground) == set(Layer) and made.parts


@pytest.mark.parametrize('recipe', [look.GRASS, look.DIRT, look.FOREST_FLOOR, look.ROCK])
def test_a_mottled_surface_is_square_in_range_and_varies(recipe):
    maps = look.mottled(recipe, 32)
    assert maps.base.shape == (32, 32, 3)
    for layer in (maps.base, maps.roughness, maps.height):
        assert 0.0 <= layer.min() and layer.max() <= 1.0
    assert maps.base.std() > 0.01


def test_each_mottled_surface_has_its_own_colours():
    means = [look.mottled(r, 32).base.mean(axis=(0, 1))
             for r in (look.GRASS, look.DIRT, look.FOREST_FLOOR, look.ROCK)]
    assert len({tuple(np.round(m, 2)) for m in means}) == 4
