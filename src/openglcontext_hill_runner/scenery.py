"""The hill as meshes: ground, trails, obstacles, hedges, rocks and trees. No GL.

Every function takes the :class:`~openglcontext_hill_runner.landscape.Landscape` it dresses
and returns :class:`~OpenGLContext.scenegraph.surfaces.Geometry` ready for the tile generator
to wear a material. The hill runs along -Z, so a point at distance ``d`` has ``z = origin - d``:
world coordinates where ``origin`` is 0, and measured from the tile's own origin where it is the
distance of the chunk's start, which keeps the 32-bit coordinates of a tile far down the hill
small and so precise. Textures are laid by the same measure, so ``origin`` should be a whole
number of texture repeats; chunk starts are. Nothing here holds state: a chunk is dressed the same way each
time it is asked for, and neighbouring chunks meet exactly because the land is one function
of position and the ground mesh is laid on one lattice.

Obstacle sizes come from the constants in :mod:`~openglcontext_hill_runner.course`: a log is as
high as a jump must clear, and the opening under a gate is as high as a slide fits.
"""
import enum
import math
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
from OpenGLContext.scenegraph import natural
from OpenGLContext.scenegraph import surfaces as sf
from OpenGLContext.scenegraph.surfaces import Geometry

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner.landscape import Landscape

#: The land reaches this far to either side of the middle of the hill, in metres.
HALF_WIDTH = 100.0
PATH_WIDTH = 0.5
#: Metres between the samples of the height grid the ground cover is seated from, and how many
#: chunks of it are kept.
FIELD_STEP = 0.5
GRID_CHUNKS = 12
#: Metres between rows of the ground mesh along the hill, and between its columns where the
#: trails can be; beyond ``LATTICE_HALF`` the columns spread out.
ROW_STEP = 0.5
COLUMN_STEP = 0.5
LATTICE_HALF = 60.0
#: Metres a texture repeats on the ground.
GROUND_REPEAT = 2.0
PATH_REPEAT = 1.5
#: A path lies this far above the ground it is laid on.
PATH_LIFT = 0.03
#: Rocks lie this near the trails at nearest, and are this big at most, in metres.
ROCK_NEAREST = 1.3
ROCK_SMALLEST = 0.12
ROCK_LARGEST = 0.7
ROCKS_PER_CHUNK = 46
#: The densest a country packs rocks, as a multiple of a forest's.
ROCK_DENSITY_MAX = 6.0
#: A rock bigger than this has smaller rocks lying about it.
CLUSTER_FROM = 0.3
#: How far past the edge of an obstacle the middle of the gap between its path and the next
#: lies. Anything standing beside an obstacle stands there, 0.2 m wide, and no wider.
GAP_BESIDE = c.LANE_WIDTH / 2 - c.OBSTACLE_HALF_WIDTH
HEDGE_WIDTH = 0.14
HEDGE_HEIGHT = 0.7
#: Metres between the cross-sections of a dry-stone wall.
WALL_STEP = 0.25
#: Trees of the simple kind per side per chunk, and how close to the paths the nearest may stand.
TREES_PER_SIDE = 12
TREE_NEAREST = 2.2
TREE_FARTHEST = 40.0


class Part(enum.Enum):
    """What a piece of scenery is made of; the tile generator gives each a material."""

    ROCK = 'rock'
    BARK = 'bark'
    CUT = 'cut'
    FOLIAGE = 'foliage'


# -- the land ---------------------------------------------------------------------

class HillField:
    """The land as the engine's ground cover asks for it: a height at ``(x, z)``.

    Cover is seated on hundreds of thousands of points, so heights come from a half-metre
    grid for each chunk, laid on first use and interpolated.
    """

    def __init__(self, land: Landscape) -> None:
        self.land = land
        self._grids: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._xs = np.arange(-HALF_WIDTH, HALF_WIDTH + 1e-9, FIELD_STEP)

    def _grid(self, index: int) -> np.ndarray:
        with self._lock:
            grid = self._grids.get(index)
            if grid is not None:
                self._grids.move_to_end(index)
                return grid
        d = index * c.CHUNK_LENGTH + np.arange(0.0, c.CHUNK_LENGTH + 1e-9, FIELD_STEP)
        x, dd = np.meshgrid(self._xs, d)
        grid = self.land.height(x, dd)
        with self._lock:
            self._grids[index] = grid
            while len(self._grids) > GRID_CHUNKS:
                self._grids.popitem(last=False)
        return grid

    @property
    def cached(self) -> int:
        """How many chunks of height grid are held."""
        return len(self._grids)

    def sample(self, x: "np.ndarray | float", z: "np.ndarray | float") -> np.ndarray:
        """The land's height at each ``(x, z)``."""
        x, d = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                   -np.asarray(z, dtype=np.float64))
        out = np.empty(x.shape)
        d = np.clip(d, 0.0, None)
        index = np.floor(d / c.CHUNK_LENGTH).astype(int)
        u = (np.clip(x, -HALF_WIDTH, HALF_WIDTH) + HALF_WIDTH) / FIELD_STEP
        for chunk in np.unique(index):
            here = index == chunk
            grid = self._grid(int(chunk))
            row = np.clip(d[here] - chunk * c.CHUNK_LENGTH, 0.0, c.CHUNK_LENGTH - 1e-6) / FIELD_STEP
            j, i = row.astype(int), np.minimum(u[here].astype(int), len(self._xs) - 2)
            fj, fi = row - j, u[here] - i
            out[here] = (grid[j, i] * (1 - fj) * (1 - fi) + grid[j + 1, i] * fj * (1 - fi)
                         + grid[j, i + 1] * (1 - fj) * fi + grid[j + 1, i + 1] * fj * fi)
        return out


def _columns() -> np.ndarray:
    """Mesh columns across the hill: one lattice for every chunk, so neighbours share vertices."""
    inner = np.arange(-LATTICE_HALF, LATTICE_HALF + 1e-9, COLUMN_STEP)
    outer = np.geomspace(LATTICE_HALF + 2.0, HALF_WIDTH, 8)
    return np.concatenate([-outer[::-1], inner, outer])


def _grid_indices(rows: int, columns: int) -> np.ndarray:
    """Two upward-facing triangles for each cell of a grid whose rows run toward -Z."""
    j, i = np.meshgrid(np.arange(rows - 1), np.arange(columns - 1), indexing='ij')
    a = (j * columns + i).ravel()
    b, below = a + 1, a + columns
    return np.stack([a, b, below, b, below + 1, below], axis=1).ravel().astype(np.uint32)


def _up_normals(slope_x: np.ndarray, slope_d: np.ndarray) -> np.ndarray:
    """Unit normals of a surface rising ``slope_x`` per metre across and ``slope_d`` per metre down."""
    normals = np.stack([-slope_x, np.ones_like(slope_x), slope_d], axis=-1)
    unit: np.ndarray = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return unit.astype('f')


def terrain(land: Landscape, index: int, origin: float = 0.0) -> Geometry:
    """The land of chunk ``index``, with z measured from ``origin``."""
    xs = _columns()
    rows = int(round(c.CHUNK_LENGTH / ROW_STEP)) + 1
    d = index * c.CHUNK_LENGTH + (np.arange(rows + 2) - 1) * ROW_STEP
    x, dd = np.meshgrid(xs, d)
    h = land.height(x, dd)
    slope_d = (h[2:] - h[:-2]) / (2 * ROW_STEP)
    height = h[1:-1]
    slope_x = np.gradient(height, xs, axis=1)
    dd, x = dd[1:-1], x[1:-1]
    z = origin - dd
    positions = np.stack([x, height, z], axis=-1).reshape(-1, 3).astype('f')
    texcoords = np.stack([x, z], axis=-1).reshape(-1, 2) / GROUND_REPEAT
    return Geometry(positions, _up_normals(slope_x, slope_d).reshape(-1, 3),
                    texcoords.astype('f'),
                    np.tile(np.array([1.0, 0.0, 0.0, 1.0], 'f'), (len(positions), 1)),
                    _grid_indices(rows, len(xs)))


def path(land: Landscape, lane: int, index: int, origin: float = 0.0) -> "Geometry | None":
    """The strip of path along ``lane`` through chunk ``index``, laid just above the land; None
    where the path does not run through the chunk. z is measured from ``origin``."""
    rows = int(round(c.CHUNK_LENGTH / ROW_STEP)) + 1
    d = index * c.CHUNK_LENGTH + np.arange(rows) * ROW_STEP
    positions = land.course.lane_positions(d)[c.ALL_LANES.index(lane)]
    there = np.isfinite(positions)
    if there.sum() < 2:
        return None
    d, middle = d[there], positions[there]
    width = np.full(len(d), PATH_WIDTH)
    if lane not in c.TRUNK:
        width = width * land.course.fork_growth(d)
    x = middle[:, None] + width[:, None] * np.array([-0.5, 0.0, 0.5])[None, :]
    dd = np.repeat(d[:, None], 3, axis=1)
    height = land.height(x, dd)
    slope_d = np.gradient(height, d, axis=0)
    slope_x = np.zeros_like(height)
    z = origin - dd
    positions3 = np.stack([x, height + PATH_LIFT, z], axis=-1).reshape(-1, 3).astype('f')
    texcoords = np.stack([x - middle[:, None], z], axis=-1).reshape(-1, 2) / PATH_REPEAT
    return Geometry(positions3, _up_normals(slope_x, slope_d).reshape(-1, 3),
                    texcoords.astype('f'),
                    np.tile(np.array([1.0, 0.0, 0.0, 1.0], 'f'), (len(positions3), 1)),
                    _grid_indices(len(d), 3))


# -- what stands on the hill ---------------------------------------------------------

def obstacle(land: Landscape, thing: c.Obstacle, origin: float = 0.0) -> dict[Part, Geometry]:
    """The pieces of an obstacle, by what each is made of.

    It stands across the paths it blocks, turned to the way the trail runs there, and fills
    the width of them: a fallen trunk or a boulder for a low obstacle, a fence of rails on two
    posts for a high one, a stack of logs or a pile of rock for a wall. Which of the two each
    kind is comes from where it stands. z is measured from ``origin``.
    """
    if thing.kind is c.Kind.CREVICE:
        return {}                         # a crevice is cut into the ground itself, not stood on it
    course = land.course
    d = thing.distance
    xs = [course.lane_x(lane, d) for lane in thing.lanes]
    x = (xs[0] + xs[-1]) / 2
    width = (xs[-1] - xs[0]) + 2 * c.OBSTACLE_HALF_WIDTH
    ground = float(land.height(x, d))
    turn = -math.atan((course.lane_x(thing.lane, d + 1.0) - course.lane_x(thing.lane, d - 1.0)) / 2.0)
    rng = c.stream(course.seed, 'obstacle', int(d * 10), thing.lane + 2, thing.span)
    natural_stone = rng.random() < 0.5
    stand = ((x, ground, origin - d), (0.0, 1.0, 0.0, turn))

    if thing.kind is c.Kind.LOW:
        if natural_stone:
            return {Part.ROCK: sf.placed(_boulders(rng, width, c.LOW_TOP, c.DEPTH[c.Kind.LOW],
                                                   thing.span), *stand)}
        return _placed(_log(rng, width), stand)
    if thing.kind is c.Kind.HIGH:
        return _placed(_fence(rng, width), stand)
    if natural_stone:
        return {Part.ROCK: sf.placed(_boulders(rng, width + 2 * GAP_BESIDE, c.WALL_TOP, c.DEPTH[c.Kind.WALL],
                                               thing.span + 1, pile=True), *stand)}
    return _placed(_woodpile(rng, width), stand)


def _placed(parts: dict[Part, Geometry], stand: tuple[Any, Any]) -> dict[Part, Geometry]:
    """Every piece of ``parts`` moved and turned as ``stand`` says."""
    return {part: sf.placed(geometry, *stand) for part, geometry in parts.items()}


def _turned(timber: tuple[Geometry, Geometry], position: tuple[float, float, float],
            rotation: tuple[float, float, float, float]) -> dict[Part, list[Geometry]]:
    """A trunk's bark and ends, placed."""
    bark, ends = timber
    return {Part.BARK: [sf.placed(bark, position, rotation)],
            Part.CUT: [sf.placed(ends, position, rotation)]}


def _joined(pieces: list[dict[Part, list[Geometry]]]) -> dict[Part, Geometry]:
    """The pieces of several trunks, one mesh for each material."""
    gathered: dict[Part, list[Geometry]] = {}
    for piece in pieces:
        for part, geometries in piece.items():
            gathered.setdefault(part, []).extend(geometries)
    return {part: sf.merge(geometries) for part, geometries in gathered.items()}


def _log(rng: np.random.Generator, width: float) -> dict[Part, Geometry]:
    """A fallen trunk across the paths, its crest as high as a jump must clear."""
    radius = c.LOW_TOP / 2
    bark, ends = natural.timber(rng, width + 2 * GAP_BESIDE, radius, sides=24)
    lift = c.LOW_TOP - float(max(bark.positions[:, 1].max(), ends.positions[:, 1].max()))
    return _joined([_turned((bark, ends), (0.0, lift, 0.0), (0.0, 1.0, 0.0, 0.0))])


def _fence(rng: np.random.Generator, width: float) -> dict[Part, Geometry]:
    """Four rails lashed across two posts, from the height a slide clears to the height a jump
    reaches: slide under, because a jump meets the rails.

    The posts stand in the gaps beside the blocked paths and the rails end at them, so nothing
    reaches into the path of a runner in the next lane.
    """
    post_radius, rail_radius = 0.075, 0.15
    top = c.HIGH_TOP + 0.25
    gap = width / 2 + GAP_BESIDE
    pieces = []
    for side in (-1, 1):
        post = natural.timber(rng, top + 0.4, post_radius, rings=10, sides=10, crook=0.01)
        pieces.append(_turned(post, (side * gap, (top - 0.4) / 2, 0.0), (0.0, 0.0, 1.0, math.pi / 2)))
    rails = 4
    reach = 1.2 * rail_radius
    rise = c.HIGH_TOP - c.HIGH_BOTTOM - 2 * reach
    for k in range(rails):
        rail = natural.timber(rng, 2 * gap + 2 * post_radius, rail_radius, sides=12, crook=0.01)
        pieces.append(_turned(rail, (0.0, c.HIGH_BOTTOM + reach + k * rise / (rails - 1),
                                     (0.04 if k % 2 else -0.04)), (0.0, 1.0, 0.0, 0.0)))
    return _joined(pieces)


def _woodpile(rng: np.random.Generator, width: float) -> dict[Part, Geometry]:
    """A stack of logs seen end-on, as high as the rules say a wall is."""
    rows = 7
    radius = c.WALL_TOP / (2.0 + (rows - 1) * math.sqrt(3.0))
    depth = c.DEPTH[c.Kind.WALL]
    across = max(2, math.ceil(width / (2 * radius)))
    pieces = []
    for row in range(rows):
        count = across - row % 2
        for k in range(count):
            r = radius * rng.uniform(0.92, 1.04)
            x = (k - (count - 1) / 2) * 2 * radius
            y = radius + row * radius * math.sqrt(3.0)
            log = natural.timber(rng, depth * rng.uniform(0.9, 1.0), r, repeat=0.5, rings=7,
                                 sides=10)
            pieces.append(_turned(log, (x, y - 0.1, rng.uniform(-0.04, 0.04)),
                                  (0.0, 1.0, 0.0, math.pi / 2 + rng.uniform(-0.05, 0.05))))
    return _joined(pieces)


def _fitted(geometry: Geometry, low: np.ndarray, high: np.ndarray) -> Geometry:
    """``geometry`` stretched along each axis to fill the box from ``low`` to ``high``."""
    p = geometry.positions.astype(float)
    scale = (high - low) / np.maximum(np.ptp(p, axis=0), 1e-9)
    positions = (p - p.min(axis=0)) * scale + low
    normals = geometry.normals / scale
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    tangents = geometry.tangents.copy()
    tangents[:, :3] = tangents[:, :3] * scale
    tangents[:, :3] /= np.linalg.norm(tangents[:, :3], axis=1, keepdims=True)
    return Geometry(positions.astype('f'), normals.astype('f'), geometry.texcoords,
                    tangents.astype('f'), geometry.indices)


def _boulders(rng: np.random.Generator, width: float, top: float, depth: float, count: int,
              pile: bool = False, sunk: float = 0.12) -> Geometry:
    """Rocks across ``width``, ``depth`` deep and ``top`` high above the ground, sunk ``sunk``
    into it: one stone for each path across, or with ``pile`` a heap of ``count`` stones in
    layers."""
    low = np.array([-width / 2, -sunk, -depth / 2])
    if not pile:
        stones = []
        step = width / count
        for k in range(count):
            reach = step / 2 * rng.uniform(1.0, 1.15)
            tall = top if k == 0 else top * rng.uniform(0.9, 1.0)
            rock = natural.stone(rng, (reach, (tall + sunk) / 2, depth / 2), 3, repeat=1.0)
            centre = -width / 2 + (k + 0.5) * step
            stones.append(_fitted(rock, np.array([centre - reach, -sunk, -depth / 2]),
                                  np.array([centre + reach, tall, depth / 2])))
        return sf.merge(stones)
    layers = (0.0, 0.45, 0.75)
    stones = []
    for layer, floor in enumerate(layers):
        ceiling = layers[layer + 1] if layer + 1 < len(layers) else 1.0
        across = max(1, count - layer)
        for k in range(across):
            centre = (k + 0.5) / across - 0.5
            reach = 0.8 * width / across * rng.uniform(0.85, 1.0)
            rock = natural.stone(rng, (reach, 0.3, depth * 0.45), 3, repeat=1.0)
            bottom = floor * top - (sunk if layer == 0 else 0.2 * top)
            stones.append(_fitted(rock, np.array([centre * width - reach, bottom,
                                                  -depth * 0.5 * rng.uniform(0.8, 1.0)]),
                                  np.array([centre * width + reach, ceiling * top,
                                            depth * 0.5 * rng.uniform(0.8, 1.0)])))
    heap = sf.merge(stones)
    return _fitted(heap, low, np.array([width / 2, top, depth / 2]))


def divider(land: Landscape, hedge: c.Divider, origin: float = 0.0) -> Geometry:
    """A low dry-stone wall between two paths, following the land and the trail along its span; z
    is measured from ``origin``."""
    course = land.course
    d = np.linspace(hedge.start, hedge.end, max(2, math.ceil((hedge.end - hedge.start) / WALL_STEP)) + 1)
    middle = np.array([(course.lane_x(hedge.lane, v) + course.lane_x(hedge.lane + 1, v)) / 2
                       for v in d])
    line = np.stack([middle, land.height(middle, d), origin - d], axis=-1)
    rng = c.stream(course.seed, 'divider', hedge.lane + 2, int(hedge.start * 10))
    return natural.rubble(rng, line, HEDGE_WIDTH, HEDGE_HEIGHT)


def _cone(radius: float, height: float, base: np.ndarray, sides: int = 8) -> Geometry:
    """A cone of ``sides`` faces standing on ``base`` (a point), smooth-shaded."""
    angles = np.linspace(0.0, 2.0 * math.pi, sides + 1)
    rim = np.stack([np.cos(angles), np.zeros_like(angles), np.sin(angles)], axis=-1)
    slope = np.stack([rim[:, 0] * height, np.full_like(angles, radius), rim[:, 2] * height],
                     axis=-1)
    slope /= np.linalg.norm(slope, axis=1, keepdims=True)
    apex = np.tile(np.array([0.0, height, 0.0]), (sides + 1, 1))
    positions = np.concatenate([rim * radius, apex]) + base
    normals = np.concatenate([slope, slope])
    around = np.arange(sides + 1) / sides
    texcoords = np.concatenate([np.stack([around * 2.0, np.ones_like(around)], axis=-1),
                                np.stack([around * 2.0, np.zeros_like(around)], axis=-1)])
    tangent = np.stack([-rim[:, 2], np.zeros_like(angles), rim[:, 0], np.ones_like(angles)],
                       axis=-1)
    steps = np.arange(sides)
    indices = np.stack([steps, sides + 1 + steps, steps + 1], axis=1)
    return Geometry(positions.astype('f'), normals.astype('f'), texcoords.astype('f'),
                    np.concatenate([tangent, tangent]).astype('f'),
                    indices.ravel().astype(np.uint32))


def trees(land: Landscape, chunk: c.Chunk,
          origin: float = 0.0) -> tuple[Geometry | None, Geometry | None]:
    """The simple conifers of a chunk as ``(foliage, trunks)``, each one mesh: more where the
    country is wooded, none in sand (``None``, ``None``), and clear of every path. z is measured
    from ``origin``."""
    rng = c.stream(land.course.seed, 'trees', chunk.index)
    count = 2 * TREES_PER_SIDE * 3
    d = chunk.start + rng.uniform(0.0, c.CHUNK_LENGTH, count)
    lanes = land.course.lane_positions(d)
    pick = rng.integers(0, len(c.ALL_LANES), count)
    base = lanes[pick, np.arange(count)]
    side = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    top = rng.uniform(6.0, 13.0, count)
    spread = top * 0.2
    x = base + side * (TREE_NEAREST + spread + rng.uniform(0.0, TREE_FARTHEST - TREE_NEAREST, count))
    ok = np.isfinite(base)
    density = land.tree_density(np.where(ok, x, 0.0), d)
    keep = ok & (rng.random(count) * 2.2 < density) & (land.trail_gap(np.where(ok, x, 0.0), d)
                                                         >= TREE_NEAREST + spread)
    foliage: list[Geometry] = []
    trunks: list[Geometry] = []
    where = np.flatnonzero(keep)
    ground = land.height(x[where], d[where])
    for i, base_height in zip(where, ground, strict=True):
        root = np.array([x[i], float(base_height) - 0.3, origin - d[i]])
        trunks.append(sf.moved(sf.cylinder(0.22 + top[i] * 0.012, 2.6, sides=8, repeat=1.0),
                               (root + (0.0, 1.3, 0.0)).tolist()))
        for tier in range(3):
            lift = 1.6 + tier * top[i] * 0.22
            foliage.append(_cone(spread[i] * (1.0 - 0.25 * tier), top[i] * 0.42 - tier * 0.5,
                                 root + (0.0, lift, 0.0)))
    if not foliage:
        return None, None
    return sf.merge(foliage), sf.merge(trunks)


def _rock(rng: np.random.Generator, radius: float) -> Geometry:
    """One rock about the origin, flattened a little and turned, the finer for being the bigger."""
    divisions = 1 if radius < 0.22 else 2 if radius < 0.45 else 3
    shape = (radius, radius * rng.uniform(0.55, 0.8), radius * rng.uniform(0.8, 1.3))
    return sf.placed(natural.stone(rng, shape, divisions), (0.0, 0.0, 0.0),
                     (0.0, 1.0, 0.0, rng.uniform(0.0, 2.0 * math.pi)))


def rocks(land: Landscape, chunk: c.Chunk, origin: float = 0.0) -> Geometry | None:
    """The rocks of a chunk as one mesh: mostly hand-sized, a few boulders, none on a trail; None
    where there are none.

    There are more where the country is rocky, and bigger. z is measured from ``origin``.
    """
    rng = c.stream(land.course.seed, 'rocks', chunk.index)
    count = int(ROCKS_PER_CHUNK * ROCK_DENSITY_MAX)
    d = chunk.start + rng.uniform(0.0, c.CHUNK_LENGTH, count)
    lanes = land.course.lane_positions(d)
    base = lanes[rng.integers(0, len(c.ALL_LANES), count), np.arange(count)]
    side = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    x = base + side * (ROCK_NEAREST + ROCK_LARGEST + rng.exponential(6.0, count))
    ok = np.isfinite(base)
    x = np.where(ok, x, 0.0)
    density = land.rock_density(x, d)
    keep = ok & (rng.random(count) * ROCK_DENSITY_MAX < density) & (
        land.trail_gap(x, d) >= ROCK_NEAREST + ROCK_LARGEST)
    where = np.flatnonzero(keep)
    sizes = []
    for i in where:
        largest = ROCK_LARGEST * (1.0 + 0.5 * min(1.0, (density[i] - 1.0) / 4.0))
        sizes.append(ROCK_SMALLEST + (largest - ROCK_SMALLEST) * float(rng.random()) ** 3)
    # A big rock has smaller ones lying about its foot, as fallen rock does.
    at = [(x[i], d[i], size) for i, size in zip(where, sizes, strict=True)]
    for i, size in zip(where, sizes, strict=True):
        if size > CLUSTER_FROM:
            for _ in range(int(rng.integers(1, 4))):
                turn, reach = rng.uniform(0.0, 2.0 * math.pi), size * rng.uniform(1.0, 1.7)
                at.append((x[i] + reach * math.cos(turn), d[i] + reach * math.sin(turn),
                           size * rng.uniform(0.25, 0.55)))
    if not at:
        return None
    ex, ed, es = (np.array(column) for column in zip(*at, strict=True))
    clear = land.trail_gap(ex, ed) >= ROCK_NEAREST
    ground = land.height(ex, ed)
    parts = [sf.placed(_rock(rng, float(es[j])), (float(ex[j]), float(ground[j]) + 0.2 * float(es[j]),
                                                    float(origin - ed[j])))
             for j in range(len(es)) if clear[j]]
    return sf.merge(parts) if parts else None
