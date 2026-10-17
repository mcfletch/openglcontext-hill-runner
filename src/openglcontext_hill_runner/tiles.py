"""The hill as streamed tiles: one tile for each chunk, made as the runner nears it.

:class:`HillTiles` is the generator a
:class:`~OpenGLContext.scenegraph.generatedtiles.GeneratedTiles` node asks. The
engine says which chunks are near the camera; this gives each one's bounds, and
builds its meshes on a worker thread when the chunk is wanted. Each tile is built around
its own origin, the start of its chunk, and drawn translated there.

A tile's ground is drawn on the ground layer that most of it is, and where another
country shows in the tile its layer is drawn over the first, a hair above it, with the
vertex alpha rising from nothing to one across the blend.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.loaders.tiles3d.boundingvolume import BoxBV
from OpenGLContext.scenegraph.basenodes import Appearance, Group, PointLight, Shape
from OpenGLContext.scenegraph.pbrmesh import PBRMesh
from OpenGLContext.scenegraph.surfaces import Geometry, merge

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import scenery as s
from openglcontext_hill_runner.landscape import Landscape, Layer
from openglcontext_hill_runner.look import Look, make_look

#: How far behind the camera tiles are wanted, as a fraction of the reach ahead.
#: The camera trails the runner by a few metres and never turns round.
BEHIND = 0.25
#: Metres above the highest hill that a tile's bounds reach, and below the lowest ground, and
#: how far past its chunk along the hill: a triangle of a tunnel or of the ground about its opening
#: belongs to the chunk its middle is in and reaches a little over the edge.
HEADROOM = 60.0
OVERLAP = 8.0
DEPTH_BELOW = 6.0
#: A layer drawn over another lies this far above it, in metres.
OVERLAY_LIFT = 0.012
#: A layer is drawn over a tile where its weight reaches this anywhere.
OVERLAY_FROM = 0.04
#: A torch's light: its colour, how bright, and how far it reaches, in metres. The light is of the
#: flame and does not throw shadows: a shadow map for each torch of a cave would cost more than
#: the cave.
TORCH_COLOUR = (1.0, 0.58, 0.26)
TORCH_INTENSITY = 2.5
TORCH_RADIUS = 16.0


@dataclass(frozen=True)
class TileScene:
    """What the tile runtime takes of a generated tile: the node to draw."""

    group: Group


@dataclass(frozen=True, eq=False)
class Piece:
    """A mesh of a tile and what it wears: a ground layer, the trail (``None``) or a scenery part."""

    wears: "Layer | s.Part | None"
    geometry: Geometry
    colours: "np.ndarray | None" = None
    casts: bool = False
    blend: bool = False


@dataclass(frozen=True, eq=False)
class TileData:
    """A tile as plain data: its meshes, where its torches' lights hang, and its size in bytes.
    It holds no node, so it pickles and a worker process can send it."""

    pieces: "list[Piece]"
    lights: "list[tuple[float, float, float]]"
    size: int


def _bytes(geometry: Geometry) -> int:
    return int(sum(array.nbytes for array in geometry))


def overlay(ground: Geometry, tint: np.ndarray,
            weight: np.ndarray) -> tuple[Geometry, np.ndarray]:
    """The part of ``ground`` a layer shows in, lifted a hair, with the layer's weight as alpha."""
    triangles = ground.indices.reshape(-1, 3)
    keep = weight[triangles].max(axis=1) > 0.01
    used, inverse = np.unique(triangles[keep], return_inverse=True)
    positions = ground.positions[used].copy()
    positions[:, 1] += OVERLAY_LIFT
    lifted = Geometry(positions, ground.normals[used], ground.texcoords[used],
                      ground.tangents[used], inverse.reshape(-1).astype(np.uint32))
    colours = tint[used].copy()
    colours[:, 3] = np.clip(weight[used], 0.0, 1.0)
    return lifted, colours


def make_tile(land: Landscape, key: int, trees: bool = True) -> TileData:
    """The meshes of chunk ``key`` of ``land``, around the chunk's own origin. No materials and
    no GL, so it runs in a worker process."""
    course = land.course
    chunk = course.chunk(key)
    origin = key * c.CHUNK_LENGTH
    pieces: list[Piece] = []
    size = 0

    ground = s.terrain(land, key, origin)
    x, d = ground.positions[:, 0].astype(float), origin - ground.positions[:, 2].astype(float)
    tint = land.tint(x, d)
    layers = land.layer_weights(x, d)
    base = int(np.argmax(layers.mean(axis=1)))
    pieces.append(Piece(Layer(base), ground, tint))
    size += _bytes(ground) + tint.nbytes
    for index, layer in enumerate(Layer):
        if index == base or layers[index].max() < OVERLAY_FROM:
            continue
        lifted, colours = overlay(ground, tint, layers[index])
        pieces.append(Piece(layer, lifted, colours, blend=True))
        size += _bytes(lifted) + colours.nbytes

    strips = [p for p in (s.path(land, lane, key, origin) for lane in c.ALL_LANES) if p is not None]
    trail = merge(strips)
    trail_tint = land.tint(trail.positions[:, 0].astype(float),
                           origin - trail.positions[:, 2].astype(float))
    pieces.append(Piece(None, trail, trail_tint))
    size += _bytes(trail) + trail_tint.nbytes

    by_part: dict[s.Part, list[Geometry]] = {}
    rocks = s.rocks(land, chunk, origin)
    if rocks is not None:
        by_part[s.Part.ROCK] = [rocks]
    for obstacle in chunk.obstacles:
        for part, geometry in s.obstacle(land, obstacle, origin).items():
            by_part.setdefault(part, []).append(geometry)
    if chunk.dividers:
        by_part.setdefault(s.Part.ROCK, []).extend(s.divider(land, h, origin) for h in chunk.dividers)
    if trees:
        foliage, trunks = s.trees(land, chunk, origin)
        if foliage is not None and trunks is not None:
            by_part.setdefault(s.Part.FOLIAGE, []).append(foliage)
            by_part.setdefault(s.Part.BARK, []).append(trunks)
    lights: list[tuple[float, float, float]] = []
    for cave in course.caves_between(chunk.start, chunk.start + c.CHUNK_LENGTH):
        for part, parts in s.cave_parts(land, cave, key, origin).items():
            by_part.setdefault(part, []).extend(parts)
        lights.extend(s.cave_lights(land, cave, key, origin))
    for part, geometries in by_part.items():
        merged = merge(geometries)
        pieces.append(Piece(part, merged, casts=part is not s.Part.FLAME))
        size += _bytes(merged)
    return TileData(pieces, lights, size)


class TileMaker:
    """What a worker process holds: the landscape of a seed, and a ``make`` for its tiles."""

    def __init__(self, seed: int, trees: bool = True, caves: bool = True) -> None:
        self.land = Landscape(c.Course(seed, caves))
        self.trees = trees

    def make(self, key: int) -> TileData:
        """The data of chunk ``key``."""
        return make_tile(self.land, key, self.trees)


class HillTiles:
    """Generates the tiles of one landscape."""

    #: The factory a worker process builds its :class:`TileMaker` with, as the engine names it.
    MAKER = 'openglcontext_hill_runner.tiles:TileMaker'

    def __init__(self, land: Landscape, look: Look | None = None, trees: bool = True) -> None:
        self.land = land
        self.course = land.course
        self.look = look or make_look()
        #: Whether tiles plant their own simple trees; off where the woods are planted separately.
        self.trees = trees

    @property
    def maker(self) -> "tuple[str, tuple[int, bool, bool]]":
        """What :class:`~OpenGLContext.scenegraph.generatedtiles.GeneratedTiles` takes as
        ``maker`` to have the tiles made in worker processes."""
        return self.MAKER, (self.course.seed, self.trees, self.course.caves)

    # -- the generator protocol --
    def keys_near(self, point: np.ndarray, radius: float) -> range:
        """The chunks from a little behind ``point`` to ``radius`` metres ahead of it."""
        distance = -float(point[2])
        first = c.chunk_index(distance - radius * BEHIND)
        return range(first, c.chunk_index(distance + radius) + 1)

    def bounds(self, key: int) -> BoxBV:
        """The box that holds the land of chunk ``key``."""
        start = key * c.CHUNK_LENGTH
        top = c.ground_height(start) + HEADROOM
        bottom = c.ground_height(start + c.CHUNK_LENGTH) - DEPTH_BELOW
        half = (top - bottom) / 2.0
        return BoxBV((0.0, (top + bottom) / 2.0, -(start + c.CHUNK_LENGTH / 2.0)),
                     [(s.HALF_WIDTH, 0.0, 0.0), (0.0, half, 0.0),
                      (0.0, 0.0, c.CHUNK_LENGTH / 2.0 + OVERLAP)])

    def origin(self, key: int) -> tuple[float, float, float]:
        """The world position of chunk ``key``'s start, around which its meshes are built."""
        return (0.0, 0.0, -key * c.CHUNK_LENGTH)

    def generate(self, key: int) -> tuple[TileScene, int]:
        """The meshes of chunk ``key``, and their size in bytes. Runs on a worker thread."""
        return self.assemble(key, make_tile(self.land, key, self.trees))

    def assemble(self, _key: int, data: TileData) -> tuple[TileScene, int]:
        """The nodes of a tile from its data, wearing this generator's materials."""
        look = self.look
        nodes: list[Any] = []
        for piece in data.pieces:
            if piece.wears is None:
                material = look.path
            elif isinstance(piece.wears, Layer):
                material = look.ground[piece.wears]
            else:
                material = look.parts[piece.wears]
            nodes.append(self._node(piece.geometry, material, piece.colours,
                                    casts=piece.casts, blend=piece.blend))
        for at in data.lights:
            nodes.append(PointLight(location=at, color=TORCH_COLOUR, intensity=TORCH_INTENSITY,
                                    radius=TORCH_RADIUS, shadows=False))
        group = Group()
        group.children = nodes
        return TileScene(group), data.size

    # -- building shapes --
    @staticmethod
    def _node(geometry: Geometry, material: Any, colours: np.ndarray | None = None,
              casts: bool = False, blend: bool = False) -> Any:
        """A shape of ``geometry`` wearing ``material``, tinted by ``colours`` per vertex.

        ``casts`` makes it cast shadows; ``blend`` lets the alpha of ``colours`` show.
        """
        look = Appearance(material=material)
        if blend:
            look.alphaMode = 'BLEND'
        mesh = PBRMesh(positions=geometry.positions, normals=geometry.normals,
                       texcoords=geometry.texcoords, tangents=geometry.tangents,
                       indices=geometry.indices, colors=colours)
        node = Shape(geometry=mesh, appearance=look)
        node.castShadow = casts
        return node
