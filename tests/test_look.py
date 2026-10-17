"""The materials: procedural ones always, photographic ones where they can be had."""
import numpy as np
import pytest
from OpenGLContext.loaders import cc0
from PIL import Image

from openglcontext_hill_runner import look
from openglcontext_hill_runner.landscape import Layer
from openglcontext_hill_runner.scenery import Part


def test_every_part_of_the_scenery_has_a_material():
    made = look.make_look(size=32)
    assert set(made.parts) == set(Part)


def test_without_a_network_the_ground_is_procedural(monkeypatch):
    monkeypatch.setattr(cc0, 'try_material', lambda name: None)
    assert look.photographic('grass') is None
    made = look.make_look(size=32, photos=True)
    assert set(made.ground) == set(Layer) and made.path is not None


@pytest.fixture
def maps(tmp_path):
    paths = {}
    for kind in ('color', 'normal', 'roughness'):
        path = tmp_path / (kind + '.jpg')
        Image.fromarray(np.full((8, 8, 3), 90, np.uint8), 'RGB').save(path)
        paths[kind] = str(path)
    return paths


def test_a_photograph_is_a_material_wearing_its_maps(monkeypatch, maps):
    monkeypatch.setattr(cc0, 'try_material', lambda name: maps)
    material = look.photographic('grass')
    assert material.baseTexture and material.normalTexture and material.metallicRoughnessTexture
    assert material.metallic == 0.0


def test_a_photograph_with_no_normal_map_is_not_used(monkeypatch, maps):
    del maps['normal']
    monkeypatch.setattr(cc0, 'try_material', lambda name: maps)
    assert look.photographic('grass') is None


def test_photographs_are_used_for_the_ground_and_the_paths_when_asked_for(monkeypatch, maps):
    monkeypatch.setattr(cc0, 'try_material', lambda name: maps)
    made = look.make_look(size=32, photos=True)
    assert all(m.baseTexture for m in made.ground.values()) and made.path.baseTexture


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


def test_the_image_files_a_photograph_reads_are_closed(monkeypatch, maps):
    opened = []
    real = Image.open

    def spy(path, *args, **kwargs):
        picture = real(path, *args, **kwargs)
        opened.append(picture)
        return picture

    monkeypatch.setattr(look.Image, 'open', spy)
    monkeypatch.setattr(cc0, 'try_material', lambda name: maps)
    look.photographic('grass')
    assert opened and all(p.fp is None for p in opened)
