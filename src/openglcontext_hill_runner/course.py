"""The hill: its lanes, its slope, and what lies on it. No GL, no clock.

Distance is metres run down the hill and is the only coordinate a rule needs.
The world places it on the -Z axis (``z = -distance``) with the lanes spread
along X, so a position in the scene is ``(lane_x(lane), ground_height(d), -d)``.

A course is endless and is made a :class:`Chunk` at a time. A chunk is a pure
function of the course's seed and its index, so the tile generator that draws
the hill and the game that plays it ask for the same chunk and get the same
one, in any order, and a run can be replayed from its seed.
"""
import enum
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from OpenGLContext.noise import smoothstep

# -- the hill ---------------------------------------------------------------

LANE_COUNT = 3
#: Metres between the centres of neighbouring paths.
LANE_WIDTH = 0.7
#: Metres of hill in a chunk, which is also a tile of the streamed world.
CHUNK_LENGTH = 72.0
#: Metres of fall for each metre run, before the swell.
SLOPE = 0.18
#: The swell: a long rolling wave added to the slope so the hill is not a ramp.
SWELL = 2.4
SWELL_LENGTH = 48.0
#: No point of the hill is steeper than this, in metres of fall per metre run.
MAX_SLOPE = 0.30


def lane_offset(lane: float) -> float:
    """Where a path lies across its trail, from the trail's middle: lane 0 on the left."""
    return (lane - (LANE_COUNT - 1) / 2.0) * LANE_WIDTH


#: Every path a trail can have: the three of its own, and the two a fork adds to a side.
ALL_LANES = (-2, -1, 0, 1, 2, 3, 4)
TRUNK = (0, 1, 2)

#: The trail wanders this far to either side of a straight line, in metres, over waves a
#: couple of hundred metres long; it runs straight for WANDER_FREE metres and the wander grows
#: over the WANDER_START after.
WANDER = 7.0
WANDER_WAVELENGTH = 190.0
WANDER_FREE = 100.0
WANDER_START = 200.0

#: A fork first turns up at FORK_FIRST and then roughly every FORK_SPACING metres. It adds two
#: paths beside the trail, which draw away to FORK_SEPARATION metres past it over FORK_DIVERGE
#: metres, run on for a length in FORK_LENGTH, and come back and join.
FORK_FIRST = 420.0
FORK_SPACING = 1000.0
FORK_JITTER = 120.0
FORK_LENGTH = (380.0, 520.0)
FORK_DIVERGE = 140.0
#: Over this many metres at each end a fork's paths grow from nothing to full width.
FORK_GROW = 12.0
FORK_SEPARATION = 38.0
#: No obstacle stands in this many metres after the paths part, in this many before they join,
#: or in this many after, and the new paths hold none for the first and last stretch.
FORK_QUIET_AFTER_START = 30.0
FORK_QUIET_BEFORE_END = 30.0
FORK_QUIET_AFTER_END = 34.0
#: A hop reaches this far across, in metres; paths drawn further apart cannot be hopped between.
HOP_REACH = 1.6 * LANE_WIDTH
#: Forks and countries are worked out ahead for this many, which is further than anyone runs.
FORK_LIMIT = 3000
#: Chunks the course holds at once; the rest are made again when asked for.
CHUNKS_KEPT = 256

#: A cave first turns up at CAVE_FIRST and then roughly every CAVE_SPACING metres, for a length in
#: CAVE_LENGTH. No fork runs for CAVE_APPROACH metres before one or CAVE_AFTER after it: the paths
#: of a fork would run into the walls. Caves are worked out ahead for CAVE_LIMIT of them.
CAVE_FIRST = 1800.0
CAVE_SPACING = 2600.0
CAVE_JITTER = 500.0
CAVE_LENGTH = (240.0, 320.0)
CAVE_APPROACH = 170.0
CAVE_AFTER = 130.0
CAVE_LIMIT = 400
#: A cave's ceiling is out of reach at each end for this many metres, and goes from one height
#: to the next over CEILING_RAMP metres. It stands in stretches that are high, in CEILING_HIGH
#: metres above the ground, and stretches that are low, in CEILING_LOW, the high ones the longer.
CAVE_MOUTH = 40.0
CEILING_RAMP = 16.0
CEILING_HIGH = (4.4, 7.5)
CEILING_LOW = (2.35, 3.0)
HIGH_STRETCH = (28.0, 52.0)
LOW_STRETCH = (30.0, 55.0)


def ground_height(distance: Any) -> Any:
    """Height of the paths, in metres, at ``distance`` down the hill; a number or an array."""
    d = np.asarray(distance, dtype=np.float64)
    height = -SLOPE * d + SWELL * np.sin(d / SWELL_LENGTH)
    return float(height) if height.ndim == 0 else height


def smooth(t: Any) -> np.ndarray:
    """0 below 0, 1 above 1, and between them ``3t^2 - 2t^3``: level at both ends."""
    # Not np.clip, which takes several times as long on the few values this is mostly asked about.
    return np.asarray(smoothstep(np.minimum(np.maximum(t, 0.0), 1.0)))


def _smooth_number(t: float) -> float:
    """:func:`smooth` of one number, as a number."""
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return t * t * (3.0 - 2.0 * t)


#: Each kind of randomness draws from a stream of its own, so that two of them never share
#: numbers however their other keys happen to fall.
STREAMS = {name: number for number, name in enumerate(
    ('chunk', 'forks', 'wander', 'zones', 'branch', 'obstacle', 'trees', 'rocks', 'woods', 'divider',
     'caves', 'tunnel'), 1)}


def stream(seed: int, name: str, *index: int) -> np.random.Generator:
    """The random numbers of one kind (see :data:`STREAMS`) for a seed, at an index of it."""
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, STREAMS[name], *index])


def chunk_index(distance: float) -> int:
    """The chunk that ``distance`` falls in; the start line is in chunk 0."""
    return max(0, int(distance // CHUNK_LENGTH))


# -- the runner's body, which the obstacles are sized against -----------------

STAND_HEIGHT = 1.7
SLIDE_HEIGHT = 0.6
#: Highest the runner's feet rise in a jump.
JUMP_PEAK = 1.5
#: How high a ceiling has to be for a jump under it to be safe: a jump's peak, the runner's
#: height at it, and a hand's breadth.
JUMP_NEEDS = JUMP_PEAK + STAND_HEIGHT + 0.2
#: A jump that is wanted at a place is made over this many metres either side of it, which no
#: low ceiling may fall in.
JUMP_MARGIN = 14.0
#: How high the feet are, at least, over a crevice for the jump to clear it.
CREVICE_CLEAR = 0.3
#: Chance that a row of a cave where a jump is safe is a crevice across the floor.
CREVICE_CHANCE = 0.35

# -- what is on the hill ------------------------------------------------------

LOW_TOP = 1.0
HIGH_BOTTOM = 1.0
HIGH_TOP = 2.4
WALL_TOP = 2.4
MAX_COIN_HEIGHT = 2.6
#: How high a coin on the level hangs: low enough to read as above its own trail from the
#: camera behind and over the runner, high enough to be taken without a jump.
COIN_HEIGHT = 0.5
#: How high a jewel hangs: past the reach of a standing runner and within that of a jump.
JEWEL_HEIGHT = 2.3
#: How much of a coin's height counts when the runner reaches for it, above and below its middle.
COIN_HALF_HEIGHT = 0.25
#: A coin stays this far ahead of an obstacle in its own lane, and this far behind it, so that
#: the runner coming up behind sees the obstacle and not the coin.
COIN_CLEAR_BEFORE = 8.0
COIN_CLEAR_AFTER = 1.5


class Kind(enum.Enum):
    """What an obstacle asks of the runner."""

    #: A log or a rock: jump over it.
    LOW = 'low'
    #: A bough across the path: slide under it.
    HIGH = 'high'
    #: A boulder wall: hop to another path.
    WALL = 'wall'
    #: A gap across the floor of a cave: jump it.
    CREVICE = 'crevice'

    @property
    def band(self) -> tuple[float, float]:
        """Metres above the ground, bottom and top, that the obstacle fills."""
        return _BANDS[self]


_BANDS = {
    Kind.LOW: (0.0, LOW_TOP),
    Kind.HIGH: (HIGH_BOTTOM, HIGH_TOP),
    Kind.WALL: (0.0, WALL_TOP),
    Kind.CREVICE: (-3.0, 0.0),
}

#: How deep along the track an obstacle is, in metres.
DEPTH = {Kind.LOW: 1.0, Kind.HIGH: 0.3, Kind.WALL: 0.9, Kind.CREVICE: 1.8}
#: How wide an obstacle is across its path, from its middle to either edge.
OBSTACLE_HALF_WIDTH = 0.3


@dataclass(frozen=True)
class Obstacle:
    """Something across one or two paths that the runner has to get past."""

    kind: Kind
    lane: int
    #: Where the obstacle's middle lies, in metres down the hill.
    distance: float
    #: How many neighbouring paths it stretches across, from ``lane`` on.
    span: int = 1

    @property
    def depth(self) -> float:
        """How far the obstacle reaches along the trail, in metres."""
        return DEPTH[self.kind]

    @property
    def lanes(self) -> range:
        """The paths it blocks."""
        return range(self.lane, self.lane + self.span)


@dataclass(frozen=True)
class Coin:
    """Treasure hanging over a path: a coin, or a jewel that is worth more."""

    #: ``(chunk, number)``: a name the game keeps once the coin is taken.
    ident: tuple[int, int]
    lane: int
    distance: float
    #: Metres above the ground at the coin's centre.
    height: float
    #: Points; a gem is worth more than a coin.
    value: int


@dataclass(frozen=True)
class Divider:
    """A hedge between ``lane`` and ``lane + 1`` that no one can hop across."""

    lane: int
    start: float
    end: float


@dataclass(frozen=True)
class Fork:
    """Two paths that part from the trail on one side, draw away, and join it again."""

    index: int
    start: float
    end: float
    #: +1 for the right of the trail, -1 for the left.
    side: int

    @staticmethod
    def lanes_of(side: int) -> tuple[int, int]:
        """The paths a fork adds on ``side``."""
        return (LANE_COUNT, LANE_COUNT + 1) if side > 0 else (-1, -2)

    @property
    def lanes(self) -> tuple[int, int]:
        """The paths it adds."""
        return self.lanes_of(self.side)


@dataclass(frozen=True)
class Cave:
    """A stretch of the hill that runs under a ceiling."""

    index: int
    start: float
    end: float
    #: Where it comes out: ``'rocks'`` into more rock country, or ``'forest'``.
    exit: str


@dataclass(frozen=True)
class Chunk:
    """One chunk of the hill: what stands and what hangs in it."""

    index: int
    start: float
    obstacles: tuple[Obstacle, ...]
    coins: tuple[Coin, ...]
    dividers: tuple[Divider, ...]


# -- making a chunk -----------------------------------------------------------

#: Obstacles stand in rows this far apart, which is more than the runner needs
#: to hop from one path to the next at full speed.
ROW_SPACING = 24.0
ROW_COUNT = int(CHUNK_LENGTH // ROW_SPACING)
#: Nothing stands on the hill before this distance.
SAFE_START = 80.0
#: Obstacles come this many times thicker by the time the course is at its hardest.
HARDEST_AT_CHUNK = 60
#: A hedge covers this part of its chunk. Rows inside it hold no wall, and the
#: row after it is far enough on for a hop to clear its end.
DIVIDER_FROM = 4.0
DIVIDER_TO = 28.0
#: Chance that a chunk past the first few is hedged.
DIVIDER_CHANCE = 0.3

_KIND_WEIGHTS = ((Kind.LOW, 0.4), (Kind.HIGH, 0.3), (Kind.WALL, 0.3))
#: How likely each kind is to stretch across two paths rather than one.
_SPAN_CHANCE = {Kind.LOW: 0.4, Kind.HIGH: 0.35, Kind.WALL: 0.25}
#: How far past a low obstacle a jewel hangs, where a runner who jumped it is still in the air.
JEWEL_BEYOND = 2.2
#: Chance that a jewel hangs past a low obstacle, to reward the jump, and that a chunk has a
#: run of jewels over a path with nothing in it.
JEWEL_AFTER_LOG = 0.6
JEWEL_RUN_CHANCE = 0.3


class ForkState(NamedTuple):
    """Where each of a set of distances stands with respect to the forks, as arrays."""

    #: The number of the fork each distance falls in or after.
    number: np.ndarray
    start: np.ndarray
    end: np.ndarray
    #: +1 or -1: the side the fork's paths are on.
    side: np.ndarray
    #: Whether that fork is running at the distance.
    running: np.ndarray
    #: How far the fork's paths are drawn from the paths beside them, signed by side; 0 when
    #: no fork is running.
    separation: np.ndarray


class CaveState(NamedTuple):
    """Where each of a set of distances stands with respect to the caves, as arrays."""

    number: np.ndarray
    start: np.ndarray
    end: np.ndarray
    #: 1 where the cave comes out into rock country, 0 where into forest.
    into_rock: np.ndarray


class Course:
    """An endless hill, made from a seed."""

    def __init__(self, seed: int = 0, caves: bool = True) -> None:
        """The hill of ``seed``, with caves in it unless ``caves`` is false."""
        self.seed = int(seed)
        #: Whether the hill has caves in it.
        self.caves = bool(caves)
        self._chunks: OrderedDict[int, Chunk] = OrderedDict()
        self._lock = threading.Lock()
        self._phase = stream(self.seed, 'wander').uniform(0.0, 2.0 * math.pi, 2)
        rng = stream(self.seed, 'forks')
        k = np.arange(FORK_LIMIT)
        self._fork_start = FORK_FIRST + k * FORK_SPACING + rng.uniform(0.0, FORK_JITTER, FORK_LIMIT)
        self._fork_end = self._fork_start + rng.uniform(*FORK_LENGTH, FORK_LIMIT)
        self._fork_side = np.where(rng.random(FORK_LIMIT) < 0.5, -1, 1)
        has_caves = bool(caves)
        cave_rng = stream(self.seed, 'caves')
        k = np.arange(CAVE_LIMIT)
        self._cave_start = CAVE_FIRST + k * CAVE_SPACING + cave_rng.uniform(0.0, CAVE_JITTER, CAVE_LIMIT)
        self._cave_end = self._cave_start + cave_rng.uniform(*CAVE_LENGTH, CAVE_LIMIT)
        self._cave_exit = np.where(cave_rng.random(CAVE_LIMIT) < 0.5, 'rocks', 'forest')
        if not has_caves:
            self._cave_start = self._cave_end = np.full(CAVE_LIMIT, -1.0e9)
        self._profiles: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # A cave takes the place of a fork that would run into it.
        for start, end in zip(self._cave_start, self._cave_end, strict=True):
            runs_in = ((self._fork_end >= start - CAVE_APPROACH)
                       & (self._fork_start <= end + CAVE_AFTER))
            self._fork_start[runs_in] = self._fork_end[runs_in] = -1.0e9

    # -- the caves --
    def cave(self, k: int) -> Cave:
        """Cave number ``k``."""
        return Cave(k, float(self._cave_start[k]), float(self._cave_end[k]),
                    str(self._cave_exit[k]))

    def cave_at(self, distance: float) -> "Cave | None":
        """The cave that ``distance`` is in, if it is in one."""
        k = int((distance - CAVE_FIRST) // CAVE_SPACING)
        if distance < CAVE_FIRST or not 0 <= k < CAVE_LIMIT:
            return None
        if self._cave_start[k] <= distance < self._cave_end[k]:
            return self.cave(k)
        return None

    def caves_between(self, start: float, end: float) -> list[Cave]:
        """The caves that any of ``[start, end]`` is in."""
        first = max(0, int((start - CAVE_FIRST) // CAVE_SPACING) - 1)
        last = min(CAVE_LIMIT - 1, int((end - CAVE_FIRST) // CAVE_SPACING) + 1)
        return [self.cave(k) for k in range(first, last + 1)
                if self._cave_end[k] >= start and self._cave_start[k] <= end]

    def cave_state(self, distance: Any) -> CaveState:
        """The cave nearest at or ahead of each distance, as arrays: its number, where it
        starts and ends, and which way it comes out (1 for rock, 0 for forest). The first
        cave for a distance before it."""
        d = np.atleast_1d(np.asarray(distance, dtype=np.float64))
        k = np.clip(np.floor((d - CAVE_FIRST) / CAVE_SPACING).astype(int), 0, CAVE_LIMIT - 1)
        return CaveState(k, self._cave_start[k], self._cave_end[k],
                         (self._cave_exit[k] == 'rocks').astype(int))

    def in_cave(self, distance: float) -> bool:
        """Whether ``distance`` is under a ceiling."""
        return self.cave_at(distance) is not None

    def _profile(self, k: int) -> "tuple[np.ndarray, np.ndarray]":
        """Where each stretch of cave ``k``'s ceiling begins and how high it stands."""
        held = self._profiles.get(k)
        if held is not None:
            return held
        rng = stream(self.seed, 'caves', k)
        start, end = float(self._cave_start[k]), float(self._cave_end[k])
        last = end - (CAVE_MOUTH + CEILING_RAMP)
        begins, heights = [start], [float(rng.uniform(*CEILING_HIGH))]
        at, high = start + CAVE_MOUTH + float(rng.uniform(0.0, 20.0)), False
        while at < last:
            begins.append(at)
            heights.append(float(rng.uniform(*(CEILING_HIGH if high else CEILING_LOW))))
            at += float(rng.uniform(*(HIGH_STRETCH if high else LOW_STRETCH)))
            high = not high
        if last - begins[-1] < 1.2 * CEILING_RAMP and len(begins) > 1:
            # Too close for the ceiling to have reached the stretch before it.
            begins.pop()
            heights.pop()
        begins.append(last)
        heights.append(float(rng.uniform(*CEILING_HIGH)))
        made = (np.array(begins), np.array(heights))
        self._profiles[k] = made
        return made

    def ceiling(self, distance: Any) -> Any:
        """How high the ceiling is over the paths at ``distance``, in metres, or ``inf`` where
        there is none; a number or an array."""
        if np.ndim(distance) == 0:
            # One place, asked every few milliseconds of a run, and mostly in the open.
            at = float(distance)
            k = min(max(math.floor((at - CAVE_FIRST) / CAVE_SPACING), 0), CAVE_LIMIT - 1)
            if not self._cave_start[k] <= at < self._cave_end[k]:
                return math.inf
        d = np.asarray(distance, dtype=np.float64)
        out = np.full(d.shape, np.inf)
        flat, flat_out = np.atleast_1d(d), np.atleast_1d(out)
        k = np.clip(np.floor((flat - CAVE_FIRST) / CAVE_SPACING).astype(int), 0, CAVE_LIMIT - 1)
        inside = (flat >= self._cave_start[k]) & (flat < self._cave_end[k])
        for cave in np.unique(k[inside]):
            here = inside & (k == cave)
            begins, heights = self._profile(int(cave))
            stretch = np.clip(np.searchsorted(begins, flat[here], side='right') - 1, 0, len(begins) - 1)
            before = heights[np.maximum(stretch - 1, 0)]
            blend = smooth((flat[here] - begins[stretch]) / CEILING_RAMP)
            flat_out[here] = np.where(stretch == 0, heights[0],
                                      before + (heights[stretch] - before) * blend)
        flat_out = flat_out.reshape(d.shape)
        return float(flat_out) if flat_out.ndim == 0 else flat_out

    def low_ceiling(self, distance: Any) -> Any:
        """Whether a jump at ``distance`` would hit the ceiling."""
        return np.asarray(self.ceiling(distance)) < JUMP_NEEDS

    def jump_clear(self, distance: float) -> bool:
        """Whether a jump may be asked of the runner at ``distance``: no low ceiling lies within
        :data:`JUMP_MARGIN` metres of it, either way, for the jump to fall in."""
        if self.cave_at(distance - JUMP_MARGIN) is None and self.cave_at(distance + JUMP_MARGIN) is None \
                and self.cave_at(distance) is None:
            return True
        around = self.ceiling(np.linspace(distance - JUMP_MARGIN, distance + JUMP_MARGIN, 29))
        return bool((np.asarray(around) >= JUMP_NEEDS).all())

    # -- where the paths are --
    def centre(self, distance: "float | np.ndarray") -> Any:
        """How far the trail's middle lies to the right of a straight line down the hill."""
        d = np.asarray(distance, dtype=np.float64)
        wave = (np.sin(d / WANDER_WAVELENGTH + self._phase[0])
                + 0.5 * np.sin(d / (0.43 * WANDER_WAVELENGTH) + self._phase[1]))
        grown = smooth((d - WANDER_FREE) / WANDER_START)
        return WANDER * grown * wave / 1.5

    def fork_state(self, distance: Any) -> ForkState:
        """Where each distance stands with respect to the forks."""
        d = np.atleast_1d(np.asarray(distance, dtype=np.float64))
        k = np.clip(np.floor((d - FORK_FIRST) / FORK_SPACING).astype(int), 0, FORK_LIMIT - 1)
        start, end, side = self._fork_start[k], self._fork_end[k], self._fork_side[k]
        running = (d >= start) & (d <= end) & (d >= FORK_FIRST)
        away = smooth((d - start) / FORK_DIVERGE) * smooth((end - d) / FORK_DIVERGE)
        return ForkState(k, start, end, side, running,
                         np.where(running, side * FORK_SEPARATION * away, 0.0))

    def fork_growth(self, distance: Any) -> Any:
        """How fully grown the paths of the fork at ``distance`` are, from 0 where a fork begins
        and ends to 1 once the paths are full size; over the :data:`FORK_GROW` metres either end."""
        state = self.fork_state(distance)
        d = np.asarray(distance, dtype=np.float64)
        return smooth((d - state.start) / FORK_GROW) * smooth((state.end - d) / FORK_GROW)

    def fork_at(self, distance: float) -> "Fork | None":
        """The fork running at ``distance``, if one is."""
        k = int((distance - FORK_FIRST) // FORK_SPACING)
        if distance < FORK_FIRST or not 0 <= k < FORK_LIMIT:
            return None
        if self._fork_start[k] <= distance <= self._fork_end[k]:
            return self.fork(k)
        return None

    def fork(self, k: int) -> Fork:
        """Fork number ``k``."""
        return Fork(k, float(self._fork_start[k]), float(self._fork_end[k]),
                    int(self._fork_side[k]))

    def forks_between(self, start: float, end: float) -> list[Fork]:
        """The forks that run at any point of ``[start, end]``."""
        first = max(0, int((start - FORK_FIRST) // FORK_SPACING) - 1)
        last = min(FORK_LIMIT - 1, int((end - FORK_FIRST) // FORK_SPACING) + 1)
        return [self.fork(k) for k in range(first, last + 1)
                if self._fork_end[k] >= start and self._fork_start[k] <= end]

    def lanes_at(self, distance: float) -> tuple[int, ...]:
        """The paths there are at ``distance``, from left to right."""
        fork = self.fork_at(distance)
        if fork is None:
            return TRUNK
        return tuple(sorted((*TRUNK, *fork.lanes)))

    def lane_x(self, lane: int, distance: float) -> float:
        """Across the hill, where path ``lane`` lies at ``distance``.

        Worked out in plain numbers: it is asked for one place at a time, a dozen times a frame, and
        NumPy takes longer over one place than the arithmetic does. :meth:`centre` and
        :meth:`fork_state` give the same for a whole run of places at once.
        """
        d = float(distance)
        wave = (math.sin(d / WANDER_WAVELENGTH + float(self._phase[0]))
                + 0.5 * math.sin(d / (0.43 * WANDER_WAVELENGTH) + float(self._phase[1])))
        centre = WANDER * _smooth_number((d - WANDER_FREE) / WANDER_START) * wave / 1.5
        k = min(max(math.floor((d - FORK_FIRST) / FORK_SPACING), 0), FORK_LIMIT - 1)
        start, end = float(self._fork_start[k]), float(self._fork_end[k])
        extra = 0.0
        if d >= FORK_FIRST and start <= d <= end:
            side = int(self._fork_side[k])
            if lane in Fork.lanes_of(side):
                extra = (side * FORK_SEPARATION * (_smooth_number((d - start) / FORK_DIVERGE)
                                                   * _smooth_number((end - d) / FORK_DIVERGE)))
        return centre + lane_offset(lane) + extra

    def lane_positions(self, distance: "np.ndarray") -> np.ndarray:
        """Where every path of :data:`ALL_LANES` lies at each distance, ``inf`` where it does
        not exist: an array of shape ``(len(ALL_LANES), N)``."""
        state = self.fork_state(distance)
        sep, side = state.separation, np.where(state.running, state.side, 0)
        d = np.atleast_1d(np.asarray(distance, dtype=np.float64))
        centre = np.asarray(self.centre(d)).reshape(-1)
        out = np.empty((len(ALL_LANES), len(d)))
        for row, lane in enumerate(ALL_LANES):
            x = centre + lane_offset(lane)
            if lane in TRUNK:
                out[row] = x
            else:
                right = lane >= LANE_COUNT
                mine = (side > 0) if right else (side < 0)
                out[row] = np.where(mine, x + sep, np.inf)
        return out

    def blocks_hop(self, from_lane: int, to_lane: int, start: float, end: float) -> bool:
        """Whether a hop made between ``start`` and ``end`` cannot be made: a hedge stands
        between the two paths, or one of them is not there, or they are drawn too far apart."""
        low = min(from_lane, to_lane)
        if abs(from_lane - to_lane) != 1:
            return False
        for distance in np.linspace(start, end, 4):
            here = self.lanes_at(float(distance))
            if from_lane not in here or to_lane not in here:
                return True
            if abs(self.lane_x(from_lane, float(distance))
                   - self.lane_x(to_lane, float(distance))) > HOP_REACH:
                return True
        for index in range(chunk_index(start), chunk_index(end) + 1):
            for divider in self.chunk(index).dividers:
                if divider.lane == low and divider.start < end and divider.end > start:
                    return True
        return False

    def chunk(self, index: int) -> Chunk:
        """Chunk ``index``; the same one every time it is asked for."""
        with self._lock:
            held = self._chunks.get(index)
            if held is not None:
                self._chunks.move_to_end(index)
                return held
        made = self._make(index)
        with self._lock:
            self._chunks[index] = made
            while len(self._chunks) > CHUNKS_KEPT:
                self._chunks.popitem(last=False)
        return made

    @property
    def chunks_held(self) -> int:
        """How many chunks the course is holding."""
        return len(self._chunks)

    def obstacles_between(self, start: float, end: float) -> list[Obstacle]:
        """The obstacles whose middles lie in ``[start, end)``, nearest first."""
        found = [o for index in range(chunk_index(start), chunk_index(end) + 1)
                 for o in self.chunk(index).obstacles if start <= o.distance < end]
        return sorted(found, key=lambda o: o.distance)

    def coins_between(self, start: float, end: float) -> list[Coin]:
        """The coins lying in ``[start, end)``, nearest first."""
        found = [coin for index in range(chunk_index(start), chunk_index(end) + 1)
                 for coin in self.chunk(index).coins if start <= coin.distance < end]
        return sorted(found, key=lambda coin: coin.distance)

    # -- generation --
    def _make(self, index: int) -> Chunk:
        rng = stream(self.seed, 'chunk', index)
        start = index * CHUNK_LENGTH
        level = min(1.0, index / HARDEST_AT_CHUNK)
        hedged = index >= 3 and rng.random() < DIVIDER_CHANCE
        dividers: tuple[Divider, ...] = ()
        if hedged:
            lanes = [0, 1] if rng.random() < 0.4 else [int(rng.integers(0, 2))]
            dividers = tuple(Divider(lane, start + DIVIDER_FROM, start + DIVIDER_TO)
                             for lane in lanes)
        obstacles = self._rows(rng, start, level, hedged)
        coins = self._coins(rng, index, start, obstacles)
        return Chunk(index, start, tuple(obstacles), tuple(coins), dividers)

    def groups_at(self, distance: float, lanes: "list[int] | None" = None) -> list[list[int]]:
        """The paths at ``distance`` split into those a hop connects: neighbours drawn closer
        together than a hop reaches. Paths of a fork drawn away from the trail are a group of
        their own."""
        here = sorted(self.lanes_at(distance) if lanes is None else lanes)
        groups: list[list[int]] = []
        for lane in here:
            if groups and lane - groups[-1][-1] == 1 and (
                    self.lane_x(lane, distance) - self.lane_x(lane - 1, distance) <= HOP_REACH):
                groups[-1].append(lane)
            else:
                groups.append([lane])
        return groups

    def _quiet(self, distance: float) -> bool:
        """Whether nothing may stand here: the paths are parting or joining."""
        return any(f.start <= distance < f.start + FORK_QUIET_AFTER_START
                   or f.end - FORK_QUIET_BEFORE_END <= distance < f.end + FORK_QUIET_AFTER_END
                   for f in self.forks_between(distance - FORK_QUIET_AFTER_END,
                                               distance + FORK_QUIET_AFTER_START))

    def _row_lanes(self, distance: float) -> list[int]:
        """The paths things may stand on at ``distance``: the new paths of a fork only well
        inside it."""
        lanes = list(self.lanes_at(distance))
        fork = self.fork_at(distance)
        if fork is not None and not (fork.start + FORK_QUIET_AFTER_START <= distance
                                     <= fork.end - FORK_QUIET_BEFORE_END):
            lanes = [lane for lane in lanes if lane in TRUNK]
        return lanes

    def _rows(self, rng: np.random.Generator, start: float,
              level: float, hedged: bool) -> list[Obstacle]:
        obstacles: list[Obstacle] = []
        for row in range(ROW_COUNT):
            offset = row * ROW_SPACING
            distance = start + offset + float(rng.integers(0, 3))
            if distance < SAFE_START or self._quiet(distance):
                continue
            under = self.cave_at(distance) is not None
            clear = not under or self.jump_clear(distance)
            if under and clear and rng.random() < CREVICE_CHANCE:
                obstacles.append(Obstacle(Kind.CREVICE, 0, distance, LANE_COUNT))
                continue
            inside_hedge = hedged and DIVIDER_FROM <= offset <= DIVIDER_TO
            available = self._row_lanes(distance)
            weights = np.array([0.30 - 0.18 * level, 0.45, 0.20 + 0.10 * level,
                                0.05 + 0.08 * level])
            groups = self.groups_at(distance, available)
            filled = min(int(rng.choice(4, p=weights / weights.sum())), len(available))
            picked = sorted(available[int(i)] for i in rng.permutation(len(available))[:filled])
            covered: set[int] = set()
            walled: set[int] = set()
            for lane in picked:
                if lane in covered:
                    continue
                # Under a low ceiling the log that would want a jump is a bough to slide under.
                easy = Kind.LOW if clear else Kind.HIGH
                kind = self._kind(rng)
                if kind is Kind.LOW:
                    kind = easy
                joined = (lane + 1 in available and lane + 1 not in picked
                          and self.lane_x(lane + 1, distance) - self.lane_x(lane, distance)
                          <= HOP_REACH)
                span = 2 if joined and rng.random() < _SPAN_CHANCE[kind] else 1
                lanes = set(range(lane, lane + span))
                shut = any(set(group) <= (walled | lanes) for group in groups
                           if set(group) & lanes)
                if kind is Kind.WALL and (inside_hedge or shut):
                    kind = easy
                if kind is Kind.WALL:
                    walled |= lanes
                covered |= lanes
                obstacles.append(Obstacle(kind, lane, distance, span))
        return obstacles

    @staticmethod
    def _kind(rng: np.random.Generator) -> Kind:
        kinds = [kind for kind, _ in _KIND_WEIGHTS]
        weights = [weight for _, weight in _KIND_WEIGHTS]
        return kinds[int(rng.choice(len(kinds), p=weights))]

    def _coins(self, rng: np.random.Generator, index: int, start: float,
               obstacles: list[Obstacle]) -> list[Coin]:
        coins: list[Coin] = []

        def hidden(lane: int, distance: float) -> bool:
            """Whether a coin here would stand between the runner and an obstacle."""
            return any(lane in o.lanes
                       and o.distance - o.depth / 2 - COIN_CLEAR_BEFORE <= distance
                       <= o.distance + o.depth / 2 + COIN_CLEAR_AFTER for o in obstacles)

        def place(lane: int, distance: float, height: float, value: int) -> None:
            if height > float(self.ceiling(distance)) - 0.3 or (
                    value > 1 and not self.jump_clear(distance)):
                return
            if (start <= distance < start + CHUNK_LENGTH and lane in self._row_lanes(distance)
                    and not hidden(lane, distance) and not self._quiet(distance)):
                coins.append(Coin((index, len(coins)), lane, distance, height, value))

        def some_lane(distance: float) -> int:
            lanes = self._row_lanes(distance)
            return lanes[int(rng.integers(0, len(lanes)))]

        for _ in range(int(rng.integers(2, 4))):
            lane = some_lane(start + CHUNK_LENGTH / 2)
            first = start + float(rng.integers(2, int(CHUNK_LENGTH) - 14))
            for step in range(6):
                place(lane, first + 2.0 * step, COIN_HEIGHT, 1)
        for obstacle in obstacles:
            if obstacle.kind is Kind.LOW and rng.random() < JEWEL_AFTER_LOG:
                place(obstacle.lane, obstacle.distance + obstacle.depth / 2 + JEWEL_BEYOND,
                      JEWEL_HEIGHT, 5)
        if rng.random() < JEWEL_RUN_CHANCE:
            first = start + float(rng.integers(4, int(CHUNK_LENGTH) - 10))
            lane = some_lane(first)
            for step in range(int(rng.integers(1, 4))):
                place(lane, first + 2.5 * step, JEWEL_HEIGHT, 5)
        return coins
