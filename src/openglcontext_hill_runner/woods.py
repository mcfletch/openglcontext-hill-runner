"""The woods: the forest demo's trees, planted along the hill a window at a time.

The tree models and their far-field cards are the forest demo's art, which the
engine draws through :class:`~OpenGLContext.scenegraph.vegetation.field.VegetationField`:
real branches near the camera, cards beyond. A field holds a fixed table of
trees, and the hill has no end, so the woods keep one field over the stretch of
hill around the runner and plant the next stretch as the runner nears the end
of this one. A chunk's trees are a function of its index, so a stretch is the
same wherever it starts.

Where the art is not installed there are no woods: the tile generator plants
its own simple trees instead.
"""
import math
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, NamedTuple

import numpy as np
from OpenGLContext.loaders.tiles3d.vegetation import poisson_thin
from OpenGLContext.scenegraph.group import Group
from OpenGLContext.scenegraph.vegetation.field import TreeSpecies, VegetationField

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner.landscape import Landscape

#: How far a tree stands from the middle of the nearest path, in metres, before its crown is
#: counted: the path and a verge. A tree stands beyond that by the radius of its own crown as
#: well, so no branch hangs over a runner.
CLEARING = 0.9
#: How much wider than its card a tree's crown reaches, counting the boughs that droop.
CROWN_REACH = 1.15
FARTHEST = 46.0
#: The least that two trees stand apart, in metres. Each keeps a disc to itself that grows with
#: its height and shrinks where the country is thick, so a tall tree has room for its crown.
SPACING = 2.8
CANDIDATES = 240
#: The densest a country packs trees, as a multiple of a forest's.
DENSITY_MAX = 2.2
#: A window of the hill reaches this far behind the runner and ahead of it. The next window is
#: worked out once the runner is within ``PREPARE_BEFORE_END`` of the end of this one, and swapped
#: in when it is ready.
BEHIND = 128.0
AHEAD = 1600.0
PREPARE_BEFORE_END = 0.75 * AHEAD

#: What grows, as the forest demo's art names it: ``(name, mesh, trunk keys, leaf keys,
#: trunk texture, leaf texture, card, card width, root sink as a fraction of height,
#: range of heights in metres)``.
_FIR_KEYS = (('oP', 'oN', 'oU', 'oI'), ('bP', 'bN', 'bU', 'bI'))
_MAPLE_KEYS = (('bP', 'bN', 'bU', 'bI'), ('cP', 'cN', 'cU', 'cI'))
KINDS: tuple[tuple[Any, ...], ...] = (
    ('fir', 'fir.npz', *_FIR_KEYS, 'fir_bark.png', 'fir_branch.png', 'fir_imp.png',
     0.50, 0.10, (13.0, 20.0)),
    ('noel', 'noel.npz', *_FIR_KEYS, 'noel_bark.png', 'noel_branch.png', 'noel_imp.png',
     0.55, 0.02, (13.0, 20.0)),
    *(('maple%d' % k, 'maple%d.npz' % k, *_MAPLE_KEYS, 'maple_bark.png', 'maple_leaves.png',
       'maple_imp%d.png' % k, 0.72, 0.02, (9.0, 14.0)) for k in range(4)),
)
#: How likely a tree is a conifer, and which kinds those are.
CONIFER = 0.6
CONIFERS = (0, 1)
MAPLES = (2, 3, 4, 5)


def _art_directory() -> str:
    from openglcontext_hill_runner import content
    return content.forest_directory()


def forest_art() -> str | None:
    """Where the forest demo's art is, or None where it is not installed."""
    try:
        return _art_directory()
    except Exception:  # noqa: BLE001 no art, for any reason, means no woods
        return None


def species(art: str) -> list[TreeSpecies]:
    """One :class:`TreeSpecies` for each kind in :data:`KINDS`, named from the art in ``art``,
    each with the card as wide as its tree's mesh, so a tree keeps its width as it hands over
    from one to the other."""
    def at(name: str) -> str:
        return os.path.join(art, name)
    kinds = [TreeSpecies(name=name, mesh=at(mesh), solid=list(solid), foliage=list(leaf),
                         solidTexture=at(trunk), foliageTexture=at(leaves), impostor=at(card),
                         cardWidth=width)
             for name, mesh, solid, leaf, trunk, leaves, card, width, _, _ in KINDS]
    return [kind.varied(cardWidth=kind.fitted_card_width()) for kind in kinds]


class Trees(NamedTuple):
    """A stretch of woods as the table a field takes."""

    positions: np.ndarray
    yaws: np.ndarray
    heights: np.ndarray
    kinds: np.ndarray


def chunk_trees(land: Landscape, index: int) -> Trees:
    """The trees of chunk ``index``: thick in dense forest, thin in meadow and rock, none on
    sand, standing on the land beside the paths and clear of every one."""
    rng = c.stream(land.course.seed, 'woods', index)
    count = 2 * CANDIDATES
    d = index * c.CHUNK_LENGTH + rng.uniform(0.0, c.CHUNK_LENGTH, count)
    lanes = land.course.lane_positions(d)
    base = lanes[rng.integers(0, len(c.ALL_LANES), count), np.arange(count)]
    present = np.isfinite(base)
    side = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    x = np.where(present, base + side * rng.uniform(CLEARING, FARTHEST, count), 0.0)
    density = land.tree_density(x, d)
    conifer = rng.random(count) < land.conifer_share(x, d)
    kinds = np.where(conifer, rng.choice(CONIFERS, count), rng.choice(MAPLES, count))
    low = np.array([KINDS[k][-1][0] for k in kinds])
    high = np.array([KINDS[k][-1][1] for k in kinds])
    heights = (low + (high - low) * rng.random(count)).astype(np.float32)
    sink = np.array([KINDS[k][-2] for k in kinds])
    crown = np.array([KINDS[k][-3] for k in kinds]) * heights / 2.0 * CROWN_REACH
    wanted = present & (rng.random(count) * DENSITY_MAX < density) & (
        land.trail_gap(x, d) >= CLEARING + crown)
    positions = np.stack([x, land.surface(x, d) - sink * heights, -d], axis=-1).astype(np.float32)
    yaws = rng.uniform(0.0, 2.0 * math.pi, count).astype(np.float32)
    keep = np.zeros(count, bool)
    if wanted.any():
        room = (1.2 + 0.10 * heights[wanted]) / np.sqrt(np.clip(density[wanted], 0.4, DENSITY_MAX))
        keep[wanted] = poisson_thin(positions[wanted], room, priority=heights[wanted])
    return Trees(positions[keep], yaws[keep], heights[keep], kinds[keep])


def window_for(distance: float) -> tuple[int, int]:
    """``(first chunk, chunk count)`` of the stretch to plant for a runner at ``distance``."""
    first = max(0, c.chunk_index(distance - BEHIND))
    last = c.chunk_index(distance + AHEAD) + 1
    return first, last - first + 1


class Woods(Group):
    """One field of trees over the stretch of hill around the runner.

    The next stretch is worked out on a worker thread once the runner is within
    ``PREPARE_BEFORE_END`` of the end of this one, and swapped into the same field when it is
    ready, so the models and textures are made once and no frame waits for a table of trees.
    """

    def __init__(self, kinds: list[TreeSpecies], land: Landscape) -> None:
        super().__init__()
        self.kinds = kinds
        self.land = land
        self.field: VegetationField | None = None
        self.window = (0, 0)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='woods')
        self._pending: Future[tuple[tuple[int, int], tuple[np.ndarray, ...]]] | None = None

    def update(self, eye: Any, facing: Any = (0.0, 0.0, -1.0), view: Any = None) -> None:
        """Start on the next stretch when the runner nears the end of this one, swap it in when it
        is ready, and choose what is drawn for the camera at ``eye``."""
        distance = -float(eye[2])
        if self.field is None:
            self._plant(*self._table(self.land, window_for(distance)))
        elif self._pending is not None and self._pending.done():
            window, table = self._pending.result()
            self._pending = None
            self._plant(window, table)
        elif self._pending is None and distance > self._swap_at():
            self._pending = self._pool.submit(self._table, self.land, window_for(distance))
        assert self.field is not None
        self.field.update(eye, facing=facing, view=view)

    def _swap_at(self) -> float:
        first, count = self.window
        return (first + count) * c.CHUNK_LENGTH - PREPARE_BEFORE_END

    def wait(self, timeout: float = 10.0) -> None:
        """Wait for the next stretch to be worked out; the next :meth:`update` swaps it in."""
        if self._pending is not None:
            self._pending.result(timeout)

    def shutdown(self) -> None:
        """Stop the worker."""
        self._pool.shutdown(wait=False, cancel_futures=True)

    def relocate(self, land: Landscape) -> None:
        """Plant the woods of another landscape, from the start, in the same field."""
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        self.land = land
        if self.field is not None:
            self._plant(*self._table(land, window_for(0.0)))

    @staticmethod
    def _table(land: Landscape, window: tuple[int, int]) -> tuple[Any, ...]:
        """The trees of a window of chunks as arrays: positions, yaws, heights and kinds."""
        first, count = window
        tables = [chunk_trees(land, index) for index in range(first, first + count)]
        positions = np.concatenate([t.positions for t in tables])
        yaws = np.concatenate([t.yaws for t in tables])
        heights = np.concatenate([t.heights for t in tables])
        kinds = np.concatenate([t.kinds for t in tables])
        return window, (positions, yaws, heights, kinds)

    def _plant(self, window: tuple[int, int], table: tuple[np.ndarray, ...]) -> None:
        positions, yaws, heights, kinds = table
        self.window = window
        if self.field is None:
            self.field = VegetationField(positions, yaws, heights, self.kinds, species_id=kinds)
            self.children = [self.field]
        else:
            self.field.replant(positions, yaws, heights, kinds)
