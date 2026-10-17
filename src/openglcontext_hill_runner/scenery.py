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

import numpy as np
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
HEDGE_WIDTH = 0.14
HEDGE_HEIGHT = 0.7
#: Trees of the simple kind per side per chunk, and how close to the paths the nearest may stand.
TREES_PER_SIDE = 12
TREE_NEAREST = 2.2
TREE_FARTHEST = 40.0


class Part(enum.Enum):
    """What a piece of scenery is made of; the tile generator gives each a material."""

    WOOD = 'wood'
    BRICK = 'brick'
    STONE = 'stone'
    ROCK = 'rock'
    BARK = 'bark'
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
    the width of them: a log or a boulder for a low obstacle, a gate with a board hanging in
    it for a high one, a brick wall or a pile of rock for a wall. Which of the two each kind
    is comes from where it stands. z is measured from ``origin``.
    """
    course = land.course
    d = thing.distance
    xs = [course.lane_x(lane, d) for lane in thing.lanes]
    x = (xs[0] + xs[-1]) / 2
    width = (xs[-1] - xs[0]) + 2 * c.OBSTACLE_HALF_WIDTH
    ground = float(land.height(x, d))
    turn = -math.atan((course.lane_x(thing.lane, d + 1.0) - course.lane_x(thing.lane, d - 1.0)) / 2.0)
    rng = c.stream(course.seed, 'obstacle', int(d * 10), thing.lane + 2, thing.span)
    boulder = rng.random() < 0.5

    def placed(local: Geometry) -> Geometry:
        return sf.placed(local, (x, ground, origin - d), (0.0, 1.0, 0.0, turn))

    if thing.kind is c.Kind.LOW:
        if boulder:
            return {Part.ROCK: placed(_boulder(rng, width, c.LOW_TOP, c.DEPTH[c.Kind.LOW]))}
        return {Part.WOOD: placed(_log(width))}
    if thing.kind is c.Kind.HIGH:
        return {Part.WOOD: placed(_gate(width))}
    if boulder:
        return {Part.ROCK: placed(_boulder(rng, width, c.WALL_TOP, c.DEPTH[c.Kind.WALL]))}
    return {Part.BRICK: placed(sf.placed(
        sf.block((width, c.WALL_TOP + 0.4, 0.8), repeat=0.5, chamfer=0.04),
        (0.0, (c.WALL_TOP - 0.4) / 2, 0.0)))}


def _log(width: float) -> Geometry:
    """A trunk lying across the paths, as high as a jump must clear."""
    radius = c.LOW_TOP / 2
    lying = (0.0, 0.0, 1.0, math.pi / 2)
    body = sf.placed(sf.cylinder(radius, width, sides=16, repeat=0.5), (0.0, radius, 0.0), lying)
    caps = [sf.placed(sf.polygon(radius, sides=16, repeat=0.5),
                      (side * width / 2, radius, 0.0),
                      (0.0, 1.0, 0.0, side * math.pi / 2)) for side in (-1, 1)]
    return sf.merge([body, *caps])


def _lumpy_sphere(rng: np.random.Generator, sides: int, amount: float) -> tuple[Geometry, np.ndarray]:
    """A unit sphere and a factor for each of its vertices, near 1 within ``amount``.

    Vertices that share a place share a factor, so the surface stays closed.
    """
    ball = sf.sphere(1.0, sides=sides)
    _, first, inverse = np.unique(np.round(ball.positions, 3), axis=0, return_index=True,
                                  return_inverse=True)
    return ball, (1.0 + amount * (rng.random(len(first)) - 0.5))[inverse.reshape(-1)]


def _boulder(rng: np.random.Generator, width: float, top: float, depth: float,
             sunk: float = 0.12) -> Geometry:
    """A lumpy rock filling ``width`` by ``depth``, its top ``top`` above the ground."""
    ball, lumpy = _lumpy_sphere(rng, 10, 0.16)
    raw = ball.positions * lumpy[:, None]
    size = np.array([width, top + sunk, depth])
    scale = size / np.ptp(raw, axis=0)
    positions = (raw - raw.min(axis=0)) * scale + np.array([-width / 2, -sunk, -depth / 2])
    normals = ball.normals / scale
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return Geometry(positions.astype('f'), normals.astype('f'),
                    (ball.texcoords * np.array([width, top]) * 0.7).astype('f'),
                    ball.tangents, ball.indices)


def _gate(width: float) -> Geometry:
    """A gate with a board hanging in it: slide under, because a jump meets the board."""
    half = width / 2
    top = c.HIGH_TOP + 0.1
    posts = [sf.placed(sf.block((0.07, top + 0.3, 0.07), repeat=0.5),
                       (side * (half + 0.035), (top - 0.3) / 2, 0.0)) for side in (-1, 1)]
    beam = sf.placed(sf.block((width + 0.24, 0.09, 0.09), repeat=0.5),
                     (0.0, c.HIGH_TOP + 0.05, 0.0))
    board_height = c.HIGH_TOP - 0.02 - c.HIGH_BOTTOM
    board = sf.placed(sf.block((width - 0.04, board_height, 0.07), repeat=0.5),
                      (0.0, c.HIGH_BOTTOM + board_height / 2, 0.0))
    return sf.merge([*posts, beam, board])


def divider(land: Landscape, hedge: c.Divider, origin: float = 0.0) -> Geometry:
    """A low stone wall between two paths, following the land and the trail along its span; z is
    measured from ``origin``."""
    course = land.course
    count = max(1, math.ceil((hedge.end - hedge.start) / ROW_STEP))
    d = np.linspace(hedge.start, hedge.end, count + 1)
    middle = np.array([(course.lane_x(hedge.lane, v) + course.lane_x(hedge.lane + 1, v)) / 2
                       for v in d])
    ground = land.height(middle, d)
    z = origin - d
    half = HEDGE_WIDTH / 2
    faces = []
    for side in (-1, 1):                                           # the two long sides
        bottom = np.stack([middle + side * half, ground - 0.3, z], axis=-1)
        top = np.stack([middle + side * half, ground + HEDGE_HEIGHT, z], axis=-1)
        faces.append(_ribbon(bottom, top, (side, 0.0, 0.0), flip=side < 0))
    left = np.stack([middle - half, ground + HEDGE_HEIGHT, z], axis=-1)
    right = np.stack([middle + half, ground + HEDGE_HEIGHT, z], axis=-1)
    faces.append(_ribbon(left, right, (0.0, 1.0, 0.0), flip=True))
    for end, toward in ((0, 1.0), (-1, -1.0)):                     # the two ends
        low = np.stack([[middle[end] - half, middle[end] + half], [ground[end] - 0.3] * 2,
                        [z[end]] * 2], axis=-1)
        high = np.stack([[middle[end] - half, middle[end] + half], [ground[end] + HEDGE_HEIGHT] * 2,
                         [z[end]] * 2], axis=-1)
        faces.append(_ribbon(low, high, (0.0, 0.0, toward), flip=toward < 0))
    return sf.merge(faces)


def _ribbon(first: np.ndarray, second: np.ndarray, normal: tuple[float, float, float],
            flip: bool) -> Geometry:
    """A strip of quads between two polylines of equal length, all facing ``normal``."""
    count = len(first)
    positions = np.concatenate([first, second]).astype('f')
    along = np.concatenate([np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(first, axis=0), axis=1)])] * 2)
    across = np.concatenate([np.zeros(count),
                             np.full(count, float(np.linalg.norm(second[0] - first[0])))])
    texcoords = np.stack([along, -across], axis=-1).astype('f')
    steps = np.arange(count - 1)
    quads = np.stack([steps, steps + 1, count + steps + 1, steps, count + steps + 1,
                      count + steps], axis=1)
    if flip:
        quads = quads[:, [0, 2, 1, 3, 5, 4]]
    tangent = np.array([0.0, 0.0, -1.0, 1.0], 'f')
    return Geometry(positions, np.tile(np.asarray(normal, 'f'), (len(positions), 1)),
                    texcoords, np.tile(tangent, (len(positions), 1)),
                    quads.ravel().astype(np.uint32))


# -- trees -----------------------------------------------------------------------------

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


def _rock(rng: np.random.Generator, radius: float, centre: np.ndarray) -> Geometry:
    """One rock: a lumpy, flattened ball sunk a little into the ground at ``centre``."""
    ball, lumpy = _lumpy_sphere(rng, 8, 0.28)
    scale = radius * np.array([1.0, 0.62, 1.0 + 0.35 * rng.random()]) * lumpy[:, None]
    turn = rng.uniform(0.0, 2.0 * math.pi)
    cos, sin = math.cos(turn), math.sin(turn)
    positions = ball.positions * scale
    positions = np.stack([positions[:, 0] * cos + positions[:, 2] * sin, positions[:, 1],
                          -positions[:, 0] * sin + positions[:, 2] * cos], axis=-1)
    normals = ball.normals / np.maximum(scale, 1e-6)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.stack([normals[:, 0] * cos + normals[:, 2] * sin, normals[:, 1],
                        -normals[:, 0] * sin + normals[:, 2] * cos], axis=-1)
    return Geometry((positions + centre + (0.0, 0.25 * radius, 0.0)).astype('f'),
                    normals.astype('f'), (ball.texcoords * radius * 2.0).astype('f'),
                    ball.tangents, ball.indices)


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
    ground = land.height(x[where], d[where])
    parts = []
    for i, base_height in zip(where, ground, strict=True):
        largest = ROCK_LARGEST * (1.0 + 0.5 * min(1.0, (density[i] - 1.0) / 4.0))
        size = ROCK_SMALLEST + (largest - ROCK_SMALLEST) * float(rng.random()) ** 3
        parts.append(_rock(rng, size, np.array([x[i], float(base_height), origin - d[i]])))
    return sf.merge(parts) if parts else None
