"""The materials of the hill: grass, dirt, wood, stone, brick and leaves.

Made once, from the engine's procedural surfaces, and shared by every tile, so
each texture is uploaded to the GPU once however many tiles wear it.
"""
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.loaders import cc0
from OpenGLContext.scenegraph import surfaces as sf
from OpenGLContext.scenegraph.imagedata import ImageDataTexture
from OpenGLContext.scenegraph.physicalmaterial import PhysicalMaterial
from PIL import Image

from openglcontext_hill_runner.landscape import Layer
from openglcontext_hill_runner.scenery import Part

SIZE = 256


@dataclass(frozen=True)
class Look:
    """Every material the scenery wears."""

    #: The ground of each country.
    ground: dict[Layer, PhysicalMaterial]
    path: PhysicalMaterial
    parts: dict[Part, PhysicalMaterial]
    coin: PhysicalMaterial
    jewel: PhysicalMaterial


def _mix(a: tuple[float, float, float], b: tuple[float, float, float],
         t: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(a) * (1.0 - t[..., None]) + np.asarray(b) * t[..., None])


@dataclass(frozen=True)
class Mottle:
    """How one earthen surface is mottled.

    Two noise fields, a broad ``patch`` and a fine ``grain``, are weighted and
    clipped to blend ``dark`` into ``light``. Roughness and relief follow the
    grain as ``base + gain * grain``.
    """

    dark: tuple[float, float, float]
    light: tuple[float, float, float]
    patch: tuple[int, int, float]  # cells, octaves, weight
    grain: tuple[int, int, float]
    rough: tuple[float, float]
    height: tuple[float, float]
    seed: int  # the patch uses this seed and the grain the next one


def mottled(recipe: Mottle, size: int = SIZE) -> sf.Maps:
    """The maps of a surface mottled as ``recipe`` says."""
    cells, octaves, weight = recipe.patch
    patch = sf.fbm(size, cells, seed=recipe.seed, octaves=octaves)
    cells, octaves, fine = recipe.grain
    grain = sf.fbm(size, cells, seed=recipe.seed + 1, octaves=octaves)
    base = _mix(recipe.dark, recipe.light, np.clip(patch * weight + grain * fine, 0, 1))
    rough_base, rough_gain = recipe.rough
    height_base, height_gain = recipe.height
    return sf.Maps(np.clip(base, 0.0, 1.0),
                   np.clip(rough_base + rough_gain * grain, 0.0, 1.0),
                   np.zeros((size, size)),
                   np.clip(height_base + height_gain * grain, 0.0, 1.0))


#: Meadow grass: two greens mottled, with a fine blade grain.
GRASS = Mottle((0.045, 0.16, 0.035), (0.17, 0.34, 0.07), (6, 3, 0.8), (64, 2, 0.4),
               (0.82, 0.12), (0.4, 0.5), seed=31)
#: Packed earth, pale enough to read against the grass, with pebbles in it.
DIRT = Mottle((0.30, 0.22, 0.14), (0.52, 0.41, 0.28), (8, 3, 0.7), (96, 2, 0.5),
              (0.88, 0.08), (0.3, 0.6), seed=41)
#: Leaf litter and needles over dark earth.
FOREST_FLOOR = Mottle((0.10, 0.07, 0.045), (0.30, 0.17, 0.07), (10, 3, 0.6), (80, 2, 0.7),
                      (0.9, 0.08), (0.3, 0.6), seed=61)
#: Grey rock, cracked and mottled.
ROCK = Mottle((0.16, 0.16, 0.17), (0.42, 0.40, 0.38), (7, 4, 0.8), (24, 3, 0.3),
              (0.8, 0.15), (0.2, 0.8), seed=71)


def coin_maps(size: int = SIZE) -> sf.Maps:
    """A gold coin's face: a raised rim, a ring cut inside it, and a five-pointed star.

    The picture spans the coin edge to edge; outside the rim it is plain gold, which is
    what the coin's edge is textured from.
    """
    axis = np.linspace(-1.0, 1.0, size)
    x, y = np.meshgrid(axis, axis)
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    star = r < 0.58 * (0.62 + 0.38 * np.cos(5.0 * theta + math.pi))
    height = np.full((size, size), 0.5)
    height[(r > 0.76) & (r < 0.82)] = 0.18
    height[r >= 0.86] = 0.92
    height[star & (r < 0.7)] = 0.78
    height = np.clip(height + 0.04 * sf.fbm(size, 24, seed=51, octaves=2), 0.0, 1.0)
    wear = sf.fbm(size, 10, seed=52, octaves=3)
    base = np.asarray(sf.GOLD) * (0.78 + 0.28 * wear)[..., None]
    base = base * (1.0 - 0.35 * (height < 0.3)[..., None])
    rough = np.clip(0.28 + 0.2 * wear + 0.15 * (height < 0.3), 0.0, 1.0)
    return sf.Maps(np.clip(base, 0.0, 1.0), rough, np.ones((size, size)), height)


def _read(path: str, mode: str) -> Image.Image:
    """The picture at ``path`` in ``mode``, with its file closed."""
    with Image.open(path) as picture:
        return picture.convert(mode)


def photographic(name: str) -> PhysicalMaterial | None:
    """A CC0 photographic material from ambientCG, or None where it cannot be had.

    ``name`` is a key of :data:`OpenGLContext.loaders.cc0.CATALOG`. The first call
    downloads and caches the maps in the per-user cache; later calls, and calls
    with no network after that, read the cache.
    """
    maps = cc0.try_material(name)
    if not maps or 'color' not in maps or 'normal' not in maps:
        return None
    settings: dict[str, Any] = dict(
        baseTexture=ImageDataTexture(image=_read(maps['color'], 'RGB')),
        normalTexture=ImageDataTexture(image=_read(maps['normal'], 'RGB')),
        baseColor=(1.0, 1.0, 1.0), metallic=0.0, roughness=0.9)
    if 'roughness' in maps:
        rough = np.asarray(_read(maps['roughness'], 'L'))
        packed = np.zeros(rough.shape + (3,), np.uint8)
        packed[..., 1] = rough
        settings.update(metallicRoughnessTexture=ImageDataTexture(
            image=Image.fromarray(packed, 'RGB')), roughness=1.0)
    return PhysicalMaterial(**settings)


def sand_maps(size: int = SIZE) -> sf.Maps:
    """Golden sand with fine ripples."""
    grain = sf.fbm(size, 96, seed=81, octaves=2)
    ripple = 0.5 + 0.5 * np.sin(np.linspace(0, 40 * math.pi, size))[None, :] * sf.fbm(size, 5, seed=82)
    base = _mix((0.52, 0.40, 0.22), (0.72, 0.58, 0.34), np.clip(grain * 0.5 + ripple * 0.5, 0, 1))
    return sf.Maps(np.clip(base, 0.0, 1.0), np.clip(0.9 + 0.08 * grain, 0.0, 1.0),
                   np.zeros((size, size)), np.clip(0.3 + 0.5 * ripple, 0.0, 1.0))


#: The CC0 photographs for each ground layer, by ambientCG name or id.
PHOTOGRAPHS = {Layer.GRASS: 'grass', Layer.FOREST_FLOOR: 'forest_floor', Layer.ROCK: 'rock',
               Layer.SAND: 'Ground080'}


def make_look(size: int = SIZE, photos: bool = False) -> Look:
    """Build the materials, with procedural textures ``size`` pixels square.

    With ``photos`` the ground and the paths wear CC0 photographs where they can
    be had, and the procedural surfaces where they cannot.
    """
    procedural = {
        Layer.GRASS: (lambda: mottled(GRASS, size), 1.5),
        Layer.FOREST_FLOOR: (lambda: mottled(FOREST_FLOOR, size), 1.5),
        Layer.ROCK: (lambda: mottled(ROCK, size), 3.0),
        Layer.SAND: (lambda: sand_maps(size), 1.0),
    }
    ground = {}
    for layer, (maps, relief) in procedural.items():
        ground[layer] = ((photographic(PHOTOGRAPHS[layer]) if photos else None)
                         or sf.pbr_material(maps(), relief=relief, metallic=0.0))
    dirt = photographic('dirt') if photos else None
    return Look(
        ground=ground,
        path=dirt or sf.pbr_material(mottled(DIRT, size), relief=2.5, metallic=0.0),
        parts={
            Part.WOOD: sf.pbr_material(sf.wood(size, colour=(0.46, 0.28, 0.14)), metallic=0.0),
            Part.BRICK: sf.pbr_material(sf.brick(size, courses=8, bricks=4,
                                                 colour=(0.42, 0.17, 0.10)), metallic=0.0),
            Part.STONE: sf.pbr_material(sf.sandstone(size, colour=(0.62, 0.60, 0.55)),
                                        metallic=0.0),
            Part.ROCK: sf.pbr_material(sf.sandstone(size, colour=(0.30, 0.29, 0.27), seed=9),
                                       metallic=0.0),
            Part.BARK: sf.pbr_material(sf.wood(size, colour=(0.22, 0.13, 0.07), rings=14,
                                               seed=17), metallic=0.0),
            Part.FOLIAGE: PhysicalMaterial(baseColor=(0.035, 0.20, 0.07), metallic=0.0,
                                           roughness=0.85),
        },
        coin=sf.pbr_material(coin_maps(), relief=2.2),
        jewel=PhysicalMaterial(baseColor=(0.80, 0.04, 0.16), metallic=0.0, roughness=0.04,
                               emissiveColor=(0.30, 0.0, 0.05)),
    )
