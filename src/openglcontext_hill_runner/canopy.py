"""The shade the trees throw on the ground between them. No GL.

The ground cover is lit by a sun factor of its own and takes no part in the shadow maps, which are
for what a tree throws on the ground it stands on and the trail. So the cover is told where the
trees' shade lies: each chunk of the hill has a raster of it, made from the trees of that chunk
and the one before (a shadow runs down the hill, never up it) and kept.
"""
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import woods
from openglcontext_hill_runner.landscape import Landscape
from openglcontext_hill_runner.scenery import HALF_WIDTH

#: The way the sun's light travels: from behind the runner and a little to the left, so shadows
#: fall ahead along the trails.
SUN = (-0.18, -0.75, -0.62)
#: How much of the sun reaches the ground in the shade of a tree.
SHADOW = 0.3
#: Metres between the cells of a chunk's raster of shade.
CELL = 1.0
#: A crown begins this far up a tree, as a fraction of its height, and is this wide at its
#: widest, as a fraction of its height, narrowing to its tip; it is shaded in LEVELS slices.
CROWN_FROM = 0.2
CROWN_SPREAD = 0.2
LEVELS = 8
#: Rasters and tables of trees kept.
KEPT = 24


class Canopy:
    """The sun that reaches each point of the ground of ``land``, from 0 to 1.

    Call it with ``(x, z)`` as the ground cover asks for a sun factor: ``z`` is the distance down
    the hill, negated.
    """

    def __init__(self, land: Landscape) -> None:
        self.land = land
        self._rasters: OrderedDict[int, np.ndarray] = OrderedDict()
        self._tables: OrderedDict[int, woods.Trees] = OrderedDict()
        self._lock = threading.RLock()
        self._rows = int(round(c.CHUNK_LENGTH / CELL))
        self._columns = int(round(2.0 * HALF_WIDTH / CELL))

    def __call__(self, x: Any, z: Any) -> np.ndarray:
        x, d = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                   -np.asarray(z, dtype=np.float64))
        out = np.ones(x.shape)
        index = np.maximum(0, np.floor(d / c.CHUNK_LENGTH)).astype(int)
        for chunk in np.unique(index):
            here = index == chunk
            raster = self._raster(int(chunk))
            row = np.clip(np.floor((d[here] - chunk * c.CHUNK_LENGTH) / CELL).astype(int),
                          0, self._rows - 1)
            column = np.clip(np.floor((x[here] + HALF_WIDTH) / CELL).astype(int),
                             0, self._columns - 1)
            out[here] = raster[row, column]
        return out

    def _trees(self, index: int) -> woods.Trees:
        with self._lock:
            held = self._tables.get(index)
            if held is not None:
                self._tables.move_to_end(index)
                return held
        made = woods.chunk_trees(self.land, index)
        with self._lock:
            self._tables[index] = made
            while len(self._tables) > KEPT:
                self._tables.popitem(last=False)
        return made

    def _raster(self, index: int) -> np.ndarray:
        """The sun that reaches each cell of chunk ``index``, from the trees that shade it."""
        with self._lock:
            held = self._rasters.get(index)
            if held is not None:
                self._rasters.move_to_end(index)
                return held
        shade = np.ones((self._rows, self._columns))
        start = index * c.CHUNK_LENGTH
        along = SUN[2] / -SUN[1]
        across = SUN[0] / -SUN[1]
        for source in (index - 1, index):
            if source < 0:
                continue
            trees = self._trees(source)
            for position, height in zip(trees.positions, trees.heights, strict=True):
                self._stamp(shade, float(position[0]), -float(position[2]), float(height), start,
                            across, along)
        with self._lock:
            self._rasters[index] = shade
            while len(self._rasters) > KEPT:
                self._rasters.popitem(last=False)
        return shade

    def _stamp(self, shade: np.ndarray, x: float, d: float, height: float, start: float,
               across: float, along: float) -> None:
        """Shade the ground the crown of one tree throws its shadow on."""
        for level in range(LEVELS):
            t = CROWN_FROM + (1.0 - CROWN_FROM) * level / (LEVELS - 1.0)
            y = t * height
            # The slice of crown at this height lands on the ground this far along the sun's way.
            cx = x + across * y
            cd = d - along * y
            radius = (height * CROWN_SPREAD * (1.0 - 0.85 * (t - CROWN_FROM) / (1.0 - CROWN_FROM))
                      + 0.25)
            first = int(np.floor((cd - radius - start) / CELL))
            last = int(np.ceil((cd + radius - start) / CELL))
            left = int(np.floor((cx - radius + HALF_WIDTH) / CELL))
            right = int(np.ceil((cx + radius + HALF_WIDTH) / CELL))
            rows = np.arange(max(first, 0), min(last, self._rows - 1) + 1)
            columns = np.arange(max(left, 0), min(right, self._columns - 1) + 1)
            if not len(rows) or not len(columns):
                continue
            down = (start + (rows + 0.5) * CELL - cd)[:, None]
            over = ((columns + 0.5) * CELL - HALF_WIDTH - cx)[None, :]
            inside = down * down + over * over <= radius * radius
            patch = shade[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
            np.minimum(patch, np.where(inside, SHADOW, 1.0), out=patch)
