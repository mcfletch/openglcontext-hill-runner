"""The hill as streamed tiles: one tile for each chunk, made as the runner nears it.

:class:`HillTiles` is the generator a
:class:`~OpenGLContext.scenegraph.generatedtiles.GeneratedTiles` node asks. The
engine says which chunks are near the camera; this gives each one's bounds, and
builds its meshes on a worker thread when the chunk is wanted.

A tile's ground is drawn on the ground layer that most of it is, and where another
country shows in the tile its layer is drawn over the first, a hair above it, with the
vertex alpha rising from nothing to one across the blend.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.loaders.tiles3d.boundingvolume import BoxBV
from OpenGLContext.scenegraph.basenodes import Appearance, Group, Shape
from OpenGLContext.scenegraph.pbrmesh import PBRMesh
from OpenGLContext.scenegraph.surfaces import Geometry, merge, shape

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import scenery as s
from openglcontext_hill_runner.landscape import Landscape, Layer
from openglcontext_hill_runner.look import Look, make_look

#: How far behind the camera tiles are wanted, as a fraction of the reach ahead.
#: The camera trails the runner by a few metres and never turns round.
BEHIND = 0.25
#: Metres above the highest hill that a tile's bounds reach, and below the lowest ground.
HEADROOM = 60.0
DEPTH_BELOW = 6.0
#: A layer drawn over another lies this far above it, in metres.
OVERLAY_LIFT = 0.012
#: A layer is drawn over a tile where its weight reaches this anywhere.
OVERLAY_FROM = 0.04


@dataclass(frozen=True)
class TileScene:
    """What the tile runtime takes of a generated tile: a node to draw and its size."""

    group: Group
    center: tuple[float, float, float]
    radius: float


def _bytes(geometry: Geometry) -> int:
    return int(sum(array.nbytes for array in geometry))


class HillTiles:
    """Generates the tiles of one landscape."""

    def __init__(self, land: Landscape, look: Look | None = None, trees: bool = True) -> None:
        self.land = land
        self.course = land.course
        self.look = look or make_look()
        #: Whether tiles plant their own simple trees; off where the woods are planted separately.
        self.trees = trees

    # -- the generator protocol --
    def keys_near(self, point: np.ndarray, radius: float) -> range:
        """The chunks from a little behind ``point`` to ``radius`` metres ahead of it."""
        distance = -float(point[2])
        first = c.chunk_index(distance - radius * BEHIND)
        return range(first, c.chunk_index(distance + radius) + 1)

    def bounds(self, key: int) -> BoxBV:
        start = key * c.CHUNK_LENGTH
        top = c.ground_height(start) + HEADROOM
        bottom = c.ground_height(start + c.CHUNK_LENGTH) - DEPTH_BELOW
        half = (top - bottom) / 2.0
        return BoxBV((0.0, (top + bottom) / 2.0, -(start + c.CHUNK_LENGTH / 2.0)),
                     [(s.HALF_WIDTH, 0.0, 0.0), (0.0, half, 0.0),
                      (0.0, 0.0, c.CHUNK_LENGTH / 2.0)])

    def generate(self, key: int) -> tuple[TileScene, int]:
        """The meshes of chunk ``key``, and their size in bytes. Runs on a worker."""
        land, look = self.land, self.look
        chunk = self.course.chunk(key)
        nodes: list[Any] = []
        size = 0

        ground = s.terrain(land, key)
        x, d = ground.positions[:, 0].astype(float), -ground.positions[:, 2].astype(float)
        tint = land.tint(x, d)
        layers = land.layer_weights(x, d)
        base = int(np.argmax(layers.mean(axis=1)))
        nodes.append(self._mesh(ground, look.ground[Layer(base)], tint, False, False))
        size += _bytes(ground) + tint.nbytes
        for index, layer in enumerate(Layer):
            if index == base or layers[index].max() < OVERLAY_FROM:
                continue
            overlay, colours = self._overlay(ground, tint, layers[index])
            nodes.append(self._mesh(overlay, look.ground[layer], colours, False, True))
            size += _bytes(overlay) + colours.nbytes

        strips = [p for p in (s.path(land, lane, key) for lane in c.ALL_LANES) if p is not None]
        trail = merge(strips)
        trail_tint = land.tint(trail.positions[:, 0].astype(float),
                               -trail.positions[:, 2].astype(float))
        nodes.append(self._mesh(trail, look.path, trail_tint, False, False))
        size += _bytes(trail) + trail_tint.nbytes

        by_part: dict[s.Part, list[Geometry]] = {s.Part.ROCK: [s.rocks(land, chunk)]}
        for obstacle in chunk.obstacles:
            for part, geometry in s.obstacle(land, obstacle).items():
                by_part.setdefault(part, []).append(geometry)
        if chunk.dividers:
            by_part.setdefault(s.Part.STONE, []).extend(s.divider(land, h) for h in chunk.dividers)
        if self.trees:
            foliage, trunks = s.trees(land, chunk)
            by_part.setdefault(s.Part.FOLIAGE, []).append(foliage)
            by_part.setdefault(s.Part.BARK, []).append(trunks)
        for part, geometries in by_part.items():
            merged = merge(geometries)
            nodes.append(self._casting(merged, look.parts[part]))
            size += _bytes(merged)

        group = Group()
        group.children = nodes
        centre, radius = self.bounds(key).bounding_sphere()
        return (TileScene(group, (float(centre[0]), float(centre[1]), float(centre[2])), radius),
                size)

    # -- building shapes --
    @staticmethod
    def _casting(geometry: Geometry, material: Any) -> Any:
        placed = shape(geometry, material)
        placed.children[0].castShadow = True
        return placed

    @staticmethod
    def _overlay(ground: Geometry, tint: np.ndarray,
                 weight: np.ndarray) -> tuple[Geometry, np.ndarray]:
        """The part of ``ground`` a layer shows in, lifted a hair, with the layer's weight as alpha."""
        triangles = ground.indices.reshape(-1, 3)
        keep = weight[triangles].max(axis=1) > 0.01
        used, inverse = np.unique(triangles[keep], return_inverse=True)
        positions = ground.positions[used].copy()
        positions[:, 1] += OVERLAY_LIFT
        overlay = Geometry(positions, ground.normals[used], ground.texcoords[used],
                           ground.tangents[used], inverse.reshape(-1).astype(np.uint32))
        colours = tint[used].copy()
        colours[:, 3] = np.clip(weight[used], 0.0, 1.0)
        return overlay, colours

    @staticmethod
    def _mesh(geometry: Geometry, material: Any, colours: np.ndarray, casts: bool,
              blend: bool) -> Any:
        """A shape of ``geometry`` tinted by vertex colour; ``blend`` lets its alpha show."""
        look = Appearance(material=material)
        if blend:
            look.alphaMode = 'BLEND'
        mesh = PBRMesh(positions=geometry.positions, normals=geometry.normals,
                       texcoords=geometry.texcoords, tangents=geometry.tangents,
                       indices=geometry.indices, appearance=look, colors=colours)
        node = Shape(geometry=mesh, appearance=look)
        node.castShadow = casts
        return node
