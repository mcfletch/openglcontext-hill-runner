"""The tunnel of a cave: an arch swept along the trail, cut into rock, and where it meets the hill. No GL.

The valley over a cave is filled to the height of the hills beside it (``Landscape.surface``),
so the cave is a hole in a hill and not a lid over a trench. The tunnel is made of three things:

* the walls and ceiling, an arched section carried along a spline through the middle of the trail
  (:func:`opengl_extrusions.frames.path_frames` places the sections) and then cut into rock with
  :func:`~OpenGLContext.scenegraph.rockwork.scrape`: flat facets, steps and pits. The section
  stands on the floor of the trail and reaches below it, so the wall is there beside a crevice;
* the floor, a strip between the foot of the walls cut by the crevices;
* the portals, where the walls run into the surface of the hill. Each column of the section is
  cut off at the place it comes out of the ground, so the rim of a portal is exactly where
  the tunnel meets the hill, and the terrain is tessellated out to it (``scenery.terrain``).

Everything is a function of the land and the cave. Points are ``(x, y, z)`` with ``z`` the
negative of the distance down the hill, as the world has them; a tile moves them to its origin.
"""
import math
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from opengl_extrusions import catmull_rom
from opengl_extrusions.frames import path_frames
from OpenGLContext.scenegraph import natural, rockwork
from OpenGLContext.scenegraph.surfaces import Geometry, merge

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import portal
from openglcontext_hill_runner.landscape import (
    CAVE_HALF_WIDTH,
    CREVICE_DEPTH,
    FACE_LENGTH,
    Landscape,
)

#: Points over the arch, how much lower the arch is at the walls than the middle (a fraction of
#: the ceiling's height, rising as the fourth power of the way across so the lanes keep their
#: headroom), and how far the section reaches below the floor, so the wall is there where a
#: crevice cuts the floor away.
ARCH_POINTS = 21
ARCH_DROP = 0.45
DEEP = CREVICE_DEPTH + 0.5
#: Metres between sections where they follow the hill's face at either end, and elsewhere; and how
#: far outside the cave the first and last stand.
NEAR_STEP = 0.5
FAR_STEP = 1.0
OUTSIDE = 2.0
#: The spline is a control point every SPLINE_STEP metres, sampled SPLINE_SAMPLES times in each
#: span.
SPLINE_STEP = 4.0
SPLINE_SAMPLES = 16
#: Metres between rows of the floor, and the points across it.
FLOOR_STEP = 0.25
FLOOR_COLUMNS = 14
#: Metres a texture repeats on the rock, and the cuts that make the walls: the depth of the passes
#: from large facets to small pits (see :class:`~OpenGLContext.scenegraph.rockwork.Pass`).
ROCK_REPEAT = 2.0
CUTS = (
    rockwork.Pass((1.5, 3.0), (0.15, 0.40), 0.02),
    rockwork.Pass((0.7, 1.5), (0.10, 0.30), 0.04),
    rockwork.Pass((0.3, 0.6), (0.40, 0.90), 0.012, tilt=0.1),
    rockwork.Pass((0.3, 0.7), (0.04, 0.12), 0.03),
)
#: How much rougher the rock is at its grain, in metres, and how far a cut comes in from where
#: the walls meet other things.
GRAIN = 0.05
RAMP = 0.8
#: How many tunnels are kept.
KEPT = 3


@dataclass(frozen=True, eq=False)
class Tunnel:
    """The geometry of one cave's tunnel."""

    cave: c.Cave
    #: The walls: points, vertex normals (facing in) and triangles.
    wall_points: np.ndarray
    wall_normals: np.ndarray
    wall_triangles: np.ndarray
    #: The floor, the same way, facing up.
    floor_points: np.ndarray
    floor_normals: np.ndarray
    floor_triangles: np.ndarray
    #: Where the walls come out of the hill at the mouth and at the far end, from one foot of the
    #: wall up over the arch and down to the other, and the floor's first and last rows from one
    #: foot to the other.
    mouth: np.ndarray
    exit: np.ndarray
    mouth_floor: np.ndarray
    exit_floor: np.ndarray
    #: The ground tessellated out to each of them.
    mouth_ground: portal.Portal
    exit_ground: portal.Portal

    def piece(self, index: int, origin: float) -> "Geometry | None":
        """The part of the tunnel whose triangles lie in chunk ``index``, with ``z`` measured
        from ``origin``; None where it has none."""
        lo, hi = index * c.CHUNK_LENGTH, (index + 1) * c.CHUNK_LENGTH
        shift = np.array([0.0, 0.0, origin])
        parts = []
        for points, normals, triangles in ((self.wall_points, self.wall_normals, self.wall_triangles),
                                           (self.floor_points, self.floor_normals, self.floor_triangles)):
            d = -points[triangles][:, :, 2].mean(axis=1)
            chosen = triangles[(d >= lo) & (d < hi)]
            if len(chosen):
                parts.append(natural.projected(points + shift, normals, chosen, ROCK_REPEAT))
        return merge(parts) if parts else None


def half_width_at(above: float, height: float) -> float:
    """How far from the middle of the trail the wall stands, ``above`` metres over the floor, in a
    tunnel with a ceiling ``height`` metres over it: the width of the floor up to where the arch
    begins, and less above that."""
    top = height * (1.0 - ARCH_DROP)
    if above <= top:
        return CAVE_HALF_WIDTH
    reach = (min(1.0, (1.0 - min(above, height) / height) / ARCH_DROP)) ** 0.25
    return float(CAVE_HALF_WIDTH * reach)


def _stations(cave: c.Cave) -> np.ndarray:
    """The distances the sections stand at: close together over the faces at either end."""
    exact = np.array([cave.start, cave.end])
    grids = [np.arange(cave.start - OUTSIDE, cave.start + FACE_LENGTH + OUTSIDE, NEAR_STEP),
             np.arange(cave.start + FACE_LENGTH + OUTSIDE, cave.end - FACE_LENGTH - OUTSIDE, FAR_STEP),
             np.arange(cave.end - FACE_LENGTH - OUTSIDE, cave.end + OUTSIDE + 1e-9, NEAR_STEP)]
    d = np.concatenate(grids)
    d = d[np.min(np.abs(d[:, None] - exact[None, :]), axis=1) > 1e-3]
    return np.sort(np.concatenate([d, exact]))


def _frames(land: Landscape, d: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """The middle of the trail at each of ``d`` along a spline through it, and the direction across
    it there, ``(x, z)`` each."""
    control = np.arange(d[0] - SPLINE_STEP, d[-1] + 2.0 * SPLINE_STEP, SPLINE_STEP)
    spline = catmull_rom(np.stack([np.asarray(land.course.centre(control)), np.zeros_like(control),
                                   -control], axis=-1), samples=SPLINE_SAMPLES)
    x = np.interp(d, -spline[:, 2], spline[:, 0])
    path = np.stack([x, np.zeros_like(d), -d], axis=-1)
    frames = path_frames(path, up=(0.0, 1.0, 0.0))
    return path, frames.right


def _sections(height: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """The sections of the tunnel, ``(stations, points)`` each: how far across and how far above the
    floor each point of each section is, for ceilings ``height`` metres over the floor."""
    u = np.linspace(-1.0, 1.0, ARCH_POINTS)[1:-1]
    arch_y = height[:, None] * (1.0 - ARCH_DROP * u[None, :] ** 4)
    top = height * (1.0 - ARCH_DROP)
    zero = np.zeros_like(height)
    left = np.stack([np.full_like(height, -DEEP), zero, 0.5 * top, top], axis=1)
    right = np.stack([top, 0.5 * top, zero, np.full_like(height, -DEEP)], axis=1)
    y = np.concatenate([left, arch_y, right], axis=1)
    x = np.concatenate([np.full(4, -CAVE_HALF_WIDTH), u * CAVE_HALF_WIDTH,
                        np.full(4, CAVE_HALF_WIDTH)])
    return np.broadcast_to(x, y.shape), y


def _placed(land: Landscape, cave: c.Cave, d: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Where every point of every section is, ``(stations, points, 3)``, how far above the floor it
    is, and how far the hill stands above the floor there."""
    path, right = _frames(land, d)
    height = np.asarray(land.course.ceiling(np.clip(d, cave.start, cave.end - 1e-3)))
    across, above = _sections(height)
    x = path[:, None, 0] + across * right[:, None, 0]
    z = path[:, None, 2] + across * right[:, None, 2]
    floor = np.asarray(land.footing(x, -z))
    points = np.stack([x, floor + above, z], axis=-1)
    rise = np.asarray(land.surface(x, -z)) - floor
    return points, above, rise


@dataclass(frozen=True)
class _Bored:
    """The sections of a tunnel cut off where they come out of the hill."""

    #: The points that stay and the points the cut leaves at the mouth and the far end, as vertices,
    #: and the triangles over them.
    vertices: np.ndarray
    triangles: np.ndarray
    #: The section each vertex is in, how far above the floor it is, and whether it is a cut.
    rows: np.ndarray
    above: np.ndarray
    cut: np.ndarray
    #: Each column's first and last section that stays, and the cuts, ``(columns, 3)``.
    first: np.ndarray
    last: np.ndarray
    mouth: np.ndarray
    far: np.ndarray


def _clipped(points: np.ndarray, above: np.ndarray, rise: np.ndarray) -> _Bored:
    """The sections cut off where each point comes out of the hill, and the triangles joining them.

    A point is in the hill where the hill stands higher over the floor than it does, and a column
    of points runs from the place it goes into the hill to the place it comes out: the cut is
    where the column crosses the surface, found between two sections. The sections before it
    collapse onto it, so the triangles between them have no area and are dropped.
    """
    stations, count = above.shape
    gap = rise - np.maximum(above, 0.0)
    inside = gap > 1e-7
    first = np.argmax(inside, axis=0)
    last = stations - 1 - np.argmax(inside[::-1], axis=0)
    columns = np.arange(count)
    if not inside.any(axis=0).all() or (first == 0).any() or (last == stations - 1).any():
        raise ValueError('a column of the tunnel does not come out of the hill at both ends')
    behind = gap[first - 1, columns]
    before = np.clip(-behind / (gap[first, columns] - behind), 0.0, 1.0)
    mouth = points[first - 1, columns] + before[:, None] * (points[first, columns] - points[first - 1, columns])
    ahead = gap[last + 1, columns]
    after = np.clip(gap[last, columns] / (gap[last, columns] - ahead), 0.0, 1.0)
    far = points[last, columns] + after[:, None] * (points[last + 1, columns] - points[last, columns])
    rows = np.arange(stations)[:, None]
    kept = (rows >= first[None, :]) & (rows <= last[None, :])
    identity = np.full((stations, count), -1, dtype=np.int64)
    identity[kept] = np.arange(kept.sum())
    grid = np.where(rows < first[None, :], kept.sum() + columns[None, :],
                    np.where(rows > last[None, :], kept.sum() + count + columns[None, :], identity))
    a, b = grid[:-1, :-1], grid[:-1, 1:]
    below, diagonal = grid[1:, :-1], grid[1:, 1:]
    triangles = np.concatenate([np.stack([a, b, below], axis=-1).reshape(-1, 3),
                                np.stack([b, diagonal, below], axis=-1).reshape(-1, 3)])
    solid = ((triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
             & (triangles[:, 0] != triangles[:, 2]))
    row_of = np.concatenate([np.nonzero(kept)[0], first - 1, last + 1])
    above_of = np.concatenate([above[kept], above[first - 1, columns], above[last + 1, columns]])
    return _Bored(np.concatenate([points[kept], mouth, far]), triangles[solid], row_of, above_of,
                  np.arange(len(row_of)) >= kept.sum(), first, last, mouth, far)


def _inward(points: np.ndarray, triangles: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """``triangles`` wound so that they face toward ``centres``, a place for each triangle."""
    normals = rockwork.face_normals(points, triangles)
    toward = np.sum(normals * (centres - points[triangles].mean(axis=1)), axis=1)
    return triangles if float(np.median(toward)) > 0.0 else triangles[:, [0, 2, 1]]


def _floor(land: Landscape, left: np.ndarray, right: np.ndarray
           ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """The floor between the foot of the left wall and the foot of the right, as points, normals,
    triangles, and its first and last rows."""
    def along(line: np.ndarray, fraction: np.ndarray) -> np.ndarray:
        length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
        at = fraction * length[-1]
        return np.stack([np.interp(at, length, line[:, k]) for k in range(3)], axis=-1)

    rows = max(2, int(math.ceil(np.linalg.norm(np.diff(left, axis=0), axis=1).sum() / FLOOR_STEP)) + 1)
    fraction = np.linspace(0.0, 1.0, rows)
    edge_a, edge_b = along(left, fraction), along(right, fraction)
    across = np.linspace(0.0, 1.0, FLOOR_COLUMNS)
    grid = edge_a[:, None, :] + across[None, :, None] * (edge_b - edge_a)[:, None, :]
    x, z = grid[..., 0], grid[..., 2]
    ground = np.asarray(land.height(x, -z))
    # The feet of the walls are where their sections put them; the floor is the land between,
    # pulled to meet them.
    drift = grid[:, [0, -1], 1] - np.asarray(land.footing(x[:, [0, -1]], -z[:, [0, -1]]))
    y = ground + drift[:, :1] * (1.0 - across[None, :]) ** 6 + drift[:, 1:] * across[None, :] ** 6
    points = np.stack([x, y, z], axis=-1)
    flat = points.reshape(-1, 3)
    cells = np.arange(rows * FLOOR_COLUMNS).reshape(rows, FLOOR_COLUMNS)
    a, b, below, diagonal = cells[:-1, :-1], cells[:-1, 1:], cells[1:, :-1], cells[1:, 1:]
    triangles = np.concatenate([np.stack([a, b, below], axis=-1).reshape(-1, 3),
                                np.stack([b, diagonal, below], axis=-1).reshape(-1, 3)])
    up = rockwork.face_normals(flat, triangles)[:, 1]
    if float(np.median(up)) < 0.0:
        triangles = triangles[:, [0, 2, 1]]
    normals = natural.smooth_normals(flat, triangles)
    return flat, normals, triangles, points[0], points[-1]


def _ripples(rng: np.random.Generator, points: np.ndarray) -> np.ndarray:
    """Smooth noise from 0 to 1 at each of ``points``."""
    return np.asarray(0.5 + 0.5 * np.clip(natural.waves(rng, points, 2.0, 12.0) / 1.5, -1.0, 1.0))


def _build(land: Landscape, cave: c.Cave, keep_clear: np.ndarray) -> Tunnel:
    d = _stations(cave)
    points, above, rise = _placed(land, cave, d)
    bored = _clipped(points, above, rise)
    sections = np.where(above[..., None] >= 0.0, points, np.nan)
    middle = np.nanmean(sections, axis=1)
    triangles = _inward(bored.vertices, bored.triangles,
                        middle[bored.rows][bored.triangles].mean(axis=1))
    held = bored.cut | (bored.above <= 1e-9)
    rng = c.stream(land.course.seed, 'tunnel', cave.index)
    walls, triangles, held = rockwork.scrape(bored.vertices, triangles, rng, CUTS, pinned=held,
                                             ramp=RAMP, keep_clear=keep_clear)
    walls = rockwork.grain(walls, triangles, rng, GRAIN, _ripples, pinned=held, ramp=RAMP)
    last = points.shape[1] - 2

    def foot(column: int) -> np.ndarray:
        line = points[bored.first[column]:bored.last[column] + 1, column]
        return np.concatenate([bored.mouth[column][None], line, bored.far[column][None]])

    flat, floor_normals, floor_triangles, mouth_floor, exit_floor = _floor(land, foot(1), foot(last))
    mouth, exit_ = bored.mouth[1:last + 1], bored.far[1:last + 1]
    return Tunnel(cave, walls, natural.smooth_normals(walls, triangles), triangles, flat,
                  floor_normals, floor_triangles, mouth, exit_, mouth_floor, exit_floor,
                  portal.make(land, mouth, mouth_floor), portal.make(land, exit_, exit_floor))


_held: "weakref.WeakKeyDictionary[Landscape, OrderedDict[int, Tunnel]]" = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def bore(land: Landscape, cave: c.Cave,
         keep_clear: "np.ndarray | Callable[[], np.ndarray] | None" = None) -> Tunnel:
    """The tunnel of ``cave``, made once and kept for as long as the landscape is: the last few
    are held. ``keep_clear`` is places on the walls, ``(K, 3)``, where nothing is cut -- where
    something is fixed to them -- or a function that answers them, called only when the tunnel
    is made."""
    with _lock:
        tunnels = _held.setdefault(land, OrderedDict())
        found = tunnels.get(cave.index)
        if found is None:
            places = keep_clear() if callable(keep_clear) else keep_clear
            clear = np.zeros((0, 3)) if places is None else np.asarray(places, dtype=float)
            found = tunnels[cave.index] = _build(land, cave, clear)
            while len(tunnels) > KEPT:
                tunnels.popitem(last=False)
        else:
            tunnels.move_to_end(cave.index)
        return found
