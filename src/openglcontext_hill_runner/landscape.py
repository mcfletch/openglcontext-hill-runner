"""The country the hill runs through, and the ground of it. No GL.

The hill passes through forest, rock, sand, meadow and dense forest, a few hundred
metres of each, in an order the seed chooses. A fork's branch runs through a country
of its own. Each country has its own settings: how steeply the land rises beside the
trails, how rough it is, how thickly trees and ground cover grow, how many rocks lie
about, and which ground it is. Where two meet they blend over 140 metres.

The land is a function of position (across the hill and down it) and nothing else:
neighbouring tiles meet without a seam, and trees, cover and the runner's feet ask it
the same questions the ground mesh does. The land is cut down to the level of every
trail and rises beside it, so a trail that wanders or forks makes its own valley.
"""
import enum
from dataclasses import dataclass
from typing import Any

import numpy as np
from OpenGLContext.noise import fbm, ridged
from OpenGLContext.scenegraph.terrain import RockRelief

from openglcontext_hill_runner import course as c

#: Across this many metres from a trail the land is level, and then it rises.
FLAT_HALF_WIDTH = 2.0
#: Metres over which two countries blend into each other.
BLEND = 140.0
#: How much further off a fork's path counts as being while it is not yet full size, in metres.
FORK_PENALTY = 2.0
#: Metres across which a fork's country gives way to the trail's.
BRANCH_BLEND = 6.0
#: Countries are worked out ahead for this many, which is further than anyone runs.
ZONE_LIMIT = 6000
#: Metres between the rows of the ground mesh, which is what the rock's faces are sampled at, and
#: how far from the middle of a trail rock stands clear, and over how many metres it rises.
GROUND_STEP = 0.5
ROCK_CLEAR = 1.0
ROCK_FADE = 1.5
#: How high the bare rock of rock country stands above the land, in metres.
ROCK_HEIGHT = 0.7
#: A cave's corridor is this wide from the trail's middle. Over the length of the cave the valley
#: is filled to the height of the hills beside it, which is higher than any ceiling by the
#: thickness of the rock over it, and the fill rises from the floor at the mouth and falls to it
#: at the far end over FACE_LENGTH metres.
CAVE_HALF_WIDTH = 2.6
FACE_LENGTH = 22.0
#: Past a cave's mouth or end the surface is this many metres from a trail, which is as far as
#: the land has risen to its banks.
FAR_FROM_TRAIL = 1000.0
#: Rock country is forced this far before a cave and the way out blends to its country over
#: OUT_BLEND metres past the end; the force fades over the CAVE_FADE before and after.
CAVE_LEAD = 200.0
CAVE_FADE = 120.0
OUT_BLEND = 100.0
OUT_HOLD = 250.0
#: A crevice is cut this deep into the floor, with edges this sharp, over the width of its
#: obstacle, and fades out across the walls.
CREVICE_DEPTH = 2.6
CREVICE_EDGE = 0.35
FIRST_ZONE = 600.0
ZONE_LENGTH = (320.0, 640.0)

#: Hummocks a few metres across and lumps the size of a hand.
HUMMOCK_METRES = 2.8
HUMMOCK_HEIGHT = 0.55
LUMP_METRES = 0.85
LUMP_HEIGHT = 0.22
#: Relief on the trails, as a fraction of what it is beside them.
TRAIL_RELIEF = 0.15
#: Patches of lush and of dry ground, a few metres across.
PATCH_METRES = 3.5


#: The metres a ground texture repeats, which the stretch of a face settles to a whole number of,
#: and over how many metres past a cave's end it does.
FACE_REPEAT = 2.0
FACE_SETTLE = 30.0
#: Points of the rule that integrates how much longer a face is than the plan under it.
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(8)


def _face_length(t: np.ndarray, steep: np.ndarray) -> np.ndarray:
    """How much longer than ``t`` of its plan a face is, as a fraction of the plan, for a face that
    rises ``steep`` metres for every metre of plan: a smooth rise over the whole of it."""
    u = t[..., None] * (0.5 + 0.5 * _NODES)
    slope = steep[..., None] * 6.0 * u * (1.0 - u)
    return np.asarray(FACE_LENGTH * t * 0.5 * np.sum(_WEIGHTS * (np.sqrt(1.0 + slope ** 2) - 1.0), axis=-1))


class Biome(enum.Enum):
    """The countries the hill runs through."""

    FOREST = 0
    ROCKS = 1
    SAND = 2
    MEADOW = 3
    DENSE = 4


class Layer(enum.Enum):
    """The ground a country is drawn on."""

    GRASS = 0
    FOREST_FLOOR = 1
    ROCK = 2
    SAND = 3


@dataclass(frozen=True)
class Country:
    """What a biome is like."""

    #: Metres the land rises beside a trail, over ``ramp`` metres.
    bank: float
    ramp: float
    #: Scale of the hummocks and lumps.
    relief: float
    #: Tree density, the ferns, nettles and flowers under them, grass and rocks, each as a
    #: multiple of a forest's.
    trees: float
    cover: float
    grass: float
    rocks: float
    #: How many of the trees are conifers.
    conifers: float
    layer: Layer
    #: Multiplies the ground's colour.
    warmth: tuple[float, float, float]


COUNTRY = {
    Biome.FOREST: Country(14.0, 24.0, 1.0, 1.0, 1.0, 0.8, 1.0, 0.6, Layer.GRASS, (0.88, 0.95, 0.82)),
    Biome.ROCKS: Country(28.0, 20.0, 1.6, 0.12, 0.2, 0.25, 6.0, 0.95, Layer.ROCK, (0.92, 0.90, 0.88)),
    Biome.SAND: Country(6.0, 40.0, 0.7, 0.0, 0.03, 0.2, 1.6, 0.5, Layer.SAND, (1.0, 0.92, 0.74)),
    Biome.MEADOW: Country(4.0, 60.0, 0.8, 0.06, 0.12, 1.0, 0.35, 0.2, Layer.GRASS, (1.0, 1.0, 0.78)),
    Biome.DENSE: Country(18.0, 22.0, 1.1, 2.2, 1.0, 0.5, 0.7, 0.7, Layer.FOREST_FLOOR,
                         (0.66, 0.78, 0.64)),
}
_BIOMES = tuple(Biome)
_LAYER_OF = np.array([COUNTRY[b].layer.value for b in _BIOMES])


@dataclass(frozen=True)
class Zone:
    """A stretch of one country down the hill."""

    biome: Biome
    start: float
    length: float


def _everywhere(x: Any, z: Any) -> Any:  # noqa: ARG001 a weight is asked for at (x, z)
    """How rocky a place is where the caller has already worked that out: wholly."""
    return np.ones_like(np.asarray(x, dtype=float))


def _mix(values: "list[float]", weights: np.ndarray) -> np.ndarray:
    """A setting of each country, blended by how much of each country is at each point."""
    mixed: np.ndarray = np.tensordot(np.array(values), weights, axes=(0, 0))
    return mixed


class Landscape:
    """The land of one course."""

    def __init__(self, course: c.Course) -> None:
        self.course = course
        #: The rock of rock country: its faces are scraped into it, as a boulder's are.
        self.rock = RockRelief(where=_everywhere, height=ROCK_HEIGHT,
                               seed=course.seed)
        rng = c.stream(course.seed, 'zones')
        sequence = list(_BIOMES)
        while len(sequence) < ZONE_LIMIT:
            nxt = [_BIOMES[int(i)] for i in rng.permutation(len(_BIOMES))]
            if nxt[0] is sequence[-1]:
                nxt.append(nxt.pop(0))
            sequence += nxt
        lengths = rng.uniform(*ZONE_LENGTH, ZONE_LIMIT)
        lengths[0] = FIRST_ZONE
        starts = np.concatenate([[0.0], np.cumsum(lengths)])
        self.zones = [Zone(sequence[i], float(starts[i]), float(lengths[i]))
                      for i in range(ZONE_LIMIT)]
        self._starts = np.concatenate([starts[:ZONE_LIMIT], [np.inf, np.inf, np.inf]])
        self._zone_biome = np.array([b.value for b in sequence[:ZONE_LIMIT]] + [0, 0, 0])
        # A fork's branch is another country than the trail's where it starts.
        forks = [course.fork(k) for k in range(c.FORK_LIMIT)]
        middles = np.array([(f.start + f.end) / 2 for f in forks])
        trunk = self._zone_biome[self._zone_index(middles)]
        # The nth of the other countries, in order: n itself below the trail's, n + 1 above.
        pick = c.stream(course.seed, 'branch').integers(0, len(_BIOMES) - 1, c.FORK_LIMIT)
        self._branch = pick + (pick >= trunk)

    # -- where the paths are --
    def trail_gap(self, x: Any, distance: Any) -> np.ndarray:
        """How far each point is from the middle of the nearest path."""
        x, d = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                   np.asarray(distance, dtype=np.float64))
        shape = x.shape
        df = d.ravel()
        positions = self.course.lane_positions(df)
        gap = np.abs(positions - x.ravel()[None, :])
        # A fork's paths count for more as they grow in, so the land does not step where they
        # begin and end: they are that much further away until they are full size.
        grown = self.course.fork_growth(df)
        for row, lane in enumerate(c.ALL_LANES):
            if lane not in c.TRUNK:
                gap[row] += (1.0 - grown) * FORK_PENALTY
        gap = np.where(np.isfinite(positions), gap, np.inf).min(axis=0)
        return gap.reshape(shape)

    # -- which country --
    def _zone_index(self, d: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self._starts, d, side='right') - 1, 0, ZONE_LIMIT - 1)

    def _trail_weights(self, d: np.ndarray) -> np.ndarray:
        half = BLEND / 2.0
        idx = self._zone_index(d)
        out = np.zeros((len(_BIOMES), len(d)))
        columns = np.arange(len(d))

        def edge(k: np.ndarray) -> np.ndarray:
            return c.smooth((d - (self._starts[k] - half)) / BLEND)

        for offset in (-1, 0, 1):
            k = idx + offset
            valid = (k >= 0) & (k < ZONE_LIMIT)
            kk = np.clip(k, 0, ZONE_LIMIT - 1)
            lower = np.where(kk == 0, 1.0, edge(kk))
            part = np.where(valid, lower - edge(kk + 1), 0.0)
            np.add.at(out, (self._zone_biome[kk], columns), part)
        return out

    def weights(self, x: Any, distance: Any) -> np.ndarray:
        """How much of each country is at each point: shape ``(len(Biome),) + x.shape``, summing
        to one."""
        x, d = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                   np.asarray(distance, dtype=np.float64))
        shape, xf, df = x.shape, x.ravel(), d.ravel()
        out = self._cave_country(self._trail_weights(df), df)
        fork = self.course.fork_state(df)
        k, start, end, side, running, sep = fork
        if running.any():
            along = c.smooth((df - start) / 60.0) * c.smooth((end - df) / 60.0)
            edge_lane = self.course.centre(df) + np.where(side > 0, c.lane_offset(2),
                                                          c.lane_offset(0))
            edge_fork = edge_lane + side * (c.LANE_WIDTH) + sep
            middle = (edge_lane + edge_fork) / 2.0
            toward = np.clip(side * (xf - middle) / BRANCH_BLEND, -1.0, 1.0)
            f = c.smooth(0.5 + 0.5 * toward) * along * running
            branch = np.zeros_like(out)
            branch[self._branch[k], np.arange(len(df))] = 1.0
            out = out * (1.0 - f) + branch * f
        return out.reshape((len(_BIOMES),) + shape)

    def _cave_country(self, out: np.ndarray, d: np.ndarray) -> np.ndarray:
        """``out``, the weights of the countries along the trail, with rock country forced on
        the way to every cave and through it, and the country the cave leads to blended in
        past its end."""
        cave = self.course.cave_state(d)
        force = (c.smooth((d - (cave.start - CAVE_LEAD)) / (CAVE_FADE))
                 * c.smooth(((cave.end + OUT_HOLD + CAVE_FADE) - d) / CAVE_FADE))
        force = np.where(d >= c.CAVE_FIRST - CAVE_LEAD, force, 0.0)
        if not force.any():
            return out
        out_of = c.smooth((d - cave.end) / OUT_BLEND)
        into = np.where(cave.into_rock > 0, Biome.ROCKS.value, Biome.FOREST.value)
        columns = np.arange(len(d))
        forced = np.zeros_like(out)
        forced[Biome.ROCKS.value] += 1.0 - out_of
        np.add.at(forced, (into, columns), out_of)
        return np.asarray(out * (1.0 - force) + forced * force)

    def hill(self, distance: Any) -> np.ndarray:
        """How much of a cave's hill stands at each distance, from 0 to 1: 0 outside the cave and
        1 more than FACE_LENGTH metres inside either end of it."""
        return np.asarray(self._hill(np.atleast_1d(np.asarray(distance, dtype=np.float64)).ravel()))

    def face_stretch(self, x: Any, distance: Any) -> np.ndarray:
        """How much further along the surface it is than along the plan from the start of the
        hill, in metres, where a cave's faces rise and fall: what a texture laid down the hill is
        moved on by so that it keeps its scale on the face. Where there is no face it is 0, and
        past a cave it settles to a whole number of ``REPEAT`` metres, so the texture agrees
        with the land beyond."""
        x, d = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                   np.asarray(distance, dtype=np.float64))
        xf, df = x.ravel(), d.ravel()
        out = np.zeros_like(df)
        near = np.flatnonzero(df >= c.CAVE_FIRST - FACE_LENGTH)
        if not len(near):
            return out.reshape(x.shape)
        cave = self.course.cave_state(df[near])
        gap = self.trail_gap(xf[near], df[near])
        steep = (self._ground(xf[near], df[near], gap + FAR_FROM_TRAIL)
                 - self._ground(xf[near], df[near], gap)) / FACE_LENGTH
        t_in = np.clip((df[near] - cave.start) / FACE_LENGTH, 0.0, 1.0)
        t_out = np.clip((cave.end - df[near]) / FACE_LENGTH, 0.0, 1.0)
        whole = _face_length(np.ones_like(steep), steep)
        total = whole + (whole - _face_length(t_out, steep)) + (_face_length(t_in, steep) - whole)
        settle = c.smooth((df[near] - cave.end) / FACE_SETTLE)
        end_gap = self.trail_gap(xf[near], cave.end)
        at_end = (self._ground(xf[near], cave.end, end_gap + FAR_FROM_TRAIL)
                  - self._ground(xf[near], cave.end, end_gap)) / FACE_LENGTH
        snapped = np.round(2.0 * _face_length(np.ones_like(at_end), at_end) / FACE_REPEAT) * FACE_REPEAT
        out[near] = total * (1.0 - settle) + snapped * settle
        return out.reshape(x.shape)

    def _hill(self, d: np.ndarray) -> np.ndarray:
        """How much of a cave's hill stands at each distance, 0 to 1."""
        cave = self.course.cave_state(d)
        rise = c.smooth((d - cave.start) / FACE_LENGTH) * c.smooth((cave.end - d) / FACE_LENGTH)
        return np.asarray(np.where(d >= c.CAVE_FIRST, rise, 0.0))

    def _densely(self, values: "list[float]", x: Any, distance: Any) -> np.ndarray:
        """How thickly something grows, blended from the countries, and none under a cave's hill."""
        x, d = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                   np.asarray(distance, dtype=np.float64))
        stands = self._hill(d.ravel())
        return self._blend(values, x, d) * (1.0 - stands.reshape(d.shape))

    def _blend(self, values: "list[float]", x: Any, distance: Any) -> np.ndarray:
        return _mix(values, self.weights(x, distance))

    def tree_density(self, x: Any, distance: Any) -> np.ndarray:
        """How thick trees grow at each point, as a multiple of a forest's."""
        return self._densely([COUNTRY[b].trees for b in _BIOMES], x, distance)

    def cover_density(self, x: Any, distance: Any) -> np.ndarray:
        """How thick the ferns, nettles and flowers grow at each point, as a multiple of a
        forest's."""
        return self._densely([COUNTRY[b].cover for b in _BIOMES], x, distance)

    def grass_density(self, x: Any, distance: Any) -> np.ndarray:
        """How thick the grass grows at each point, as a multiple of a forest's."""
        return self._densely([COUNTRY[b].grass for b in _BIOMES], x, distance)

    def rock_density(self, x: Any, distance: Any) -> np.ndarray:
        """How many rocks lie about at each point, as a multiple of a forest's."""
        return self._densely([COUNTRY[b].rocks for b in _BIOMES], x, distance)

    def conifer_share(self, x: Any, distance: Any) -> np.ndarray:
        """The share of the trees at each point that are conifers, from 0 to 1."""
        return self._blend([COUNTRY[b].conifers for b in _BIOMES], x, distance)

    def layer_weights(self, x: Any, distance: Any) -> np.ndarray:
        """How much of each ground layer is at each point: ``(len(Layer),) + x.shape``."""
        w = self.weights(x, distance)
        out = np.zeros((len(Layer),) + w.shape[1:])
        for biome in _BIOMES:
            out[_LAYER_OF[biome.value]] += w[biome.value]
        return out

    # -- the shape of the ground --
    def relief(self, x: Any, distance: Any) -> np.ndarray:
        """How far the land departs from its slope, in metres: hummocks a few metres across and
        lumps the size of a hand, worn down on the trails."""
        x, d = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                   np.asarray(distance, dtype=np.float64))
        scale = self._blend([COUNTRY[b].relief for b in _BIOMES], x, d)
        return self._relief(x, d, self.trail_gap(x, d), scale)

    @staticmethod
    def _relief(x: np.ndarray, d: np.ndarray, gap: np.ndarray, scale: np.ndarray) -> np.ndarray:
        hummocks = fbm(x / HUMMOCK_METRES + 11.0, d / HUMMOCK_METRES - 7.0, seed=31,
                       octaves=2) - 0.5
        lumps = fbm(x / LUMP_METRES + 3.0, d / LUMP_METRES + 9.0, seed=32, octaves=2) - 0.5
        worn = TRAIL_RELIEF + (1.0 - TRAIL_RELIEF) * c.smooth((gap - FLAT_HALF_WIDTH / 2) / 2.0)
        return np.asarray(worn * scale * (HUMMOCK_HEIGHT * hummocks + LUMP_HEIGHT * lumps))

    def height(self, x: Any, distance: Any, crevices: bool = True) -> np.ndarray:
        """Height of the ground at ``x`` across the hill and ``distance`` down it, as a runner
        finds it: the valley of the trails, with the crevices cut into the floor of a cave
        unless ``crevices`` is false. :meth:`surface` is the top of the hill over a cave."""
        x, d = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                   np.asarray(distance, dtype=np.float64))
        xf, df = x.ravel(), d.ravel()
        height = self._ground(xf, df, self.trail_gap(xf, df))
        if crevices:
            height -= self._crevices(xf, df)
        return np.asarray(height.reshape(x.shape))

    def surface(self, x: Any, distance: Any) -> np.ndarray:
        """Height of the outside of the hill: :meth:`height` except over a cave, where the valley
        of the trail is filled to the height of the hills beside it and the tunnel is bored
        through the fill."""
        x, d = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                   np.asarray(distance, dtype=np.float64))
        xf, df = x.ravel(), d.ravel()
        gap = self.trail_gap(xf, df)
        height = self._ground(xf, df, gap)
        stands = self._hill(df)
        if stands.any():
            filled = self._ground(xf, df, gap + FAR_FROM_TRAIL)
            height = height + stands * (filled - height)
        return np.asarray(height.reshape(x.shape))

    def _ground(self, x: np.ndarray, d: np.ndarray, gap: np.ndarray) -> np.ndarray:
        """The land at points ``gap`` metres from the nearest trail."""
        w = self.weights(x, d)
        bank = _mix([COUNTRY[b].bank for b in _BIOMES], w)
        ramp = _mix([COUNTRY[b].ramp for b in _BIOMES], w)
        scale = _mix([COUNTRY[b].relief for b in _BIOMES], w)
        away = c.smooth((gap - FLAT_HALF_WIDTH) / ramp)
        form = 0.5 + 0.9 * fbm(x * 0.021 + 3.0, d * 0.021 - 5.0, seed=21, octaves=4)
        height: np.ndarray = c.ground_height(d) + bank * away * form + self._relief(x, d, gap, scale)
        height += self._character(x, d, gap, w)
        height += self._bare_rock(x, d, gap, w)
        return height

    def footing(self, x: Any, distance: Any) -> np.ndarray:
        """Where a runner's feet stand: the height of the land with no crevice cut in it, since
        a runner crosses a crevice in the air and does not follow the floor down."""
        return self.height(x, distance, crevices=False)

    def _crevices(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """How far the floor is cut away at each point by the crevices of a cave."""
        out = np.zeros_like(x)
        stands = self._hill(d)
        if not stands.any():
            return out
        wanted = stands > 0.0
        found = [o.distance for o in self.course.obstacles_between(float(d[wanted].min()) - 3.0,
                                                                   float(d[wanted].max()) + 3.0)
                 if o.kind is c.Kind.CREVICE]
        if not found:
            return out
        half = c.DEPTH[c.Kind.CREVICE] / 2.0
        along = np.zeros_like(x)
        for at in found:
            along = np.maximum(along, c.smooth((half - np.abs(d - at)) / CREVICE_EDGE))
        across = np.abs(x - np.asarray(self.course.centre(d)))
        return np.asarray(CREVICE_DEPTH * along * (1.0 - c.smooth((across - CAVE_HALF_WIDTH) / 1.0)))

    def _bare_rock(self, x: np.ndarray, d: np.ndarray, gap: np.ndarray,
                   w: np.ndarray) -> np.ndarray:
        """Rock standing on the ground of rock country, scraped into flat faces, clear of the
        trails and the strip beside them."""
        rocky = w[Biome.ROCKS.value] * c.smooth((gap - ROCK_CLEAR) / ROCK_FADE)
        found = c.smooth((rocky - self.rock.begins) / (self.rock.whole - self.rock.begins))
        out = np.zeros_like(x)
        near = found > 0.0
        if near.any():
            out[near] = found[near] * (self.rock.lift + self.rock.rock(x[near], d[near],
                                                                       GROUND_STEP))
        return out

    @staticmethod
    def _character(x: np.ndarray, d: np.ndarray, gap: np.ndarray, w: np.ndarray) -> np.ndarray:
        """What makes sand dunes and rock country, added away from the trails."""
        apart = c.smooth((gap - 1.0) / 2.5)
        dunes = (0.55 * np.sin(d / 9.0 + 0.23 * x) + 0.28 * np.sin(d / 3.7 - 0.4 * x + 1.0)
                 + 0.3 * np.sin(x / 7.0 + d / 23.0))
        crags = 6.0 * ridged(x * 0.045 + 7.0, d * 0.045 - 2.0, seed=61, octaves=3)
        shaped: np.ndarray = apart * (w[Biome.SAND.value] * dunes + w[Biome.ROCKS.value] * crags)
        return shaped

    def tint(self, x: Any, distance: Any) -> np.ndarray:
        """RGBA multiplying the ground's texture: patches of lush and dry ground, in the colour of
        the country, ``(N, 4)``."""
        x, d = np.broadcast_arrays(np.asarray(x, dtype=np.float64).reshape(-1),
                                   np.asarray(distance, dtype=np.float64).reshape(-1))
        patch = fbm(x / PATCH_METRES + 5.0, d / PATCH_METRES + 2.0, seed=41, octaves=3)
        speckle = fbm(x / 0.6 - 4.0, d / 0.6 + 1.0, seed=42, octaves=2)
        level = np.clip(0.62 + 0.55 * patch + 0.2 * speckle - 0.2, 0.5, 1.0)
        dry = np.clip((patch - 0.55) * 2.0, 0.0, 1.0)
        base = np.stack([level, level * (1.0 - 0.14 * dry), level * (1.0 - 0.38 * dry)], axis=-1)
        w = self.weights(x, d)
        warmth = np.tensordot(w.T, np.array([COUNTRY[b].warmth for b in _BIOMES]), axes=(1, 0))
        colours = np.clip(base * warmth, 0.45, 1.0)
        return np.concatenate([colours, np.ones((len(x), 1))], axis=-1).astype('f')
