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
