"""Where a tunnel comes out of the hill: the ground tessellated out to the rim of the opening. No GL.

The ground of the hill is a lattice, one vertex every half metre. Where a tunnel is bored through
it the triangles the opening cuts are taken out, and a patch is put in their place: the lattice
nodes round a window about the opening, the rim of the opening where the tunnel's walls and floor
meet the hill, and triangles between the two. The patch is a constrained Delaunay triangulation
of the ring between the window and the rim
(:func:`opengl_extrusions.tessellate`), divided further where its triangles are long, and
every vertex on the window is a node of the lattice and every vertex on the rim a vertex of the
tunnel, so the ground, the patch and the tunnel meet without a crack.
"""
from dataclasses import dataclass

import numpy as np
from opengl_extrusions import tessellate
from OpenGLContext.scenegraph import rockwork

from openglcontext_hill_runner.landscape import Landscape

#: Metres between the nodes of the lattice the ground is made on, along the hill and across it.
LATTICE = 0.5
#: How far the window reaches beyond the opening, in metres, and how long an edge of the patch
#: may be before it is divided.
MARGIN = 2.0
LONGEST = 0.9
DIVISIONS = 3


@dataclass(frozen=True, eq=False)
class Portal:
    """The ground about the opening of a tunnel."""

    #: The window, ``(x0, x1, d0, d1)``: whole lattice cells are taken out of the ground inside it.
    window: tuple[float, float, float, float]
    #: The patch: points ``(x, y, z)`` with ``z`` the negative of the distance down the hill, the
    #: surface normals there and the triangles, facing up.
    points: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray

    def covers(self, d: float) -> bool:
        """Whether distance ``d`` down the hill is in the window."""
        return self.window[2] <= d <= self.window[3]


def _rectangle(x0: float, x1: float, d0: float, d1: float) -> np.ndarray:
    """The nodes round the edge of a window, ``(N, 2)``, in order."""
    across = np.arange(x0, x1 - 1e-9, LATTICE)
    along = np.arange(d0, d1 - 1e-9, LATTICE)
    return np.concatenate([np.stack([across, np.full_like(across, d0)], axis=1),
                           np.stack([np.full_like(along, x1), along], axis=1),
                           np.stack([x1 - (across - x0), np.full_like(across, d1)], axis=1),
                           np.stack([np.full_like(along, x0), d1 - (along - d0)], axis=1)])


def _normals(land: Landscape, x: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Unit normals of the surface at each ``(x, d)``, from its slopes over a lattice step each way."""
    h = LATTICE
    slope_x = (np.asarray(land.surface(x + h, d)) - np.asarray(land.surface(x - h, d))) / (2 * h)
    slope_d = (np.asarray(land.surface(x, d + h)) - np.asarray(land.surface(x, d - h))) / (2 * h)
    normals = np.stack([-slope_x, np.ones_like(slope_x), slope_d], axis=-1)
    return np.asarray(normals / np.linalg.norm(normals, axis=-1, keepdims=True))


def make(land: Landscape, rim: np.ndarray, floor: np.ndarray) -> Portal:
    """The ground about the opening whose walls come out of the hill at ``rim`` (``(K, 3)``, from the
    foot of one wall over the arch to the foot of the other) and whose floor begins at ``floor``
    (``(F, 3)``, from one foot to the other)."""
    loop = np.concatenate([rim, floor[::-1][1:-1]])
    plan = np.stack([loop[:, 0], -loop[:, 2]], axis=1)
    low, high = plan.min(axis=0) - MARGIN, plan.max(axis=0) + MARGIN
    x0, d0 = np.floor(low / LATTICE) * LATTICE
    x1, d1 = np.ceil(high / LATTICE) * LATTICE
    outer = _rectangle(float(x0), float(x1), float(d0), float(d1))
    made = tessellate([outer, plan])
    sources = made.source_index
    ring = len(outer) + len(plan)
    where = np.full(ring, -1, dtype=np.int64)
    where[sources[sources >= 0]] = np.flatnonzero(sources >= 0)
    around = [np.stack([where[start + np.arange(count)],
                        where[start + (np.arange(count) + 1) % count]], axis=1)
              for start, count in ((0, len(outer)), (len(outer), len(plan)))]
    flat, triangles = np.concatenate([made.points, np.zeros((len(made.points), 1))], axis=1), made.triangles.astype(np.int64)
    protected = np.concatenate(around)
    for _ in range(DIVISIONS):
        flat, triangles, parents = rockwork.split_edges(
            flat, triangles, lambda a, b: np.linalg.norm(a - b, axis=1) > LONGEST, protected=protected)
        if not len(parents):
            break
    plan2 = flat[:, :2]
    y = np.asarray(land.surface(plan2[:, 0], plan2[:, 1]))
    rim_at = where[len(outer):]
    y[rim_at] = loop[:, 1]
    points = np.stack([plan2[:, 0], y, -plan2[:, 1]], axis=-1)
    normals = _normals(land, plan2[:, 0], plan2[:, 1])
    up = rockwork.face_normals(points, triangles)[:, 1]
    if float(np.median(up)) < 0.0:
        triangles = triangles[:, [0, 2, 1]]
    return Portal((float(x0), float(x1), float(d0), float(d1)), points, normals, triangles)
