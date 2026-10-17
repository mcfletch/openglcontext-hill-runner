"""The runner's body: how far it has run, how fast, and where it is in the air.

It moves down the hill by itself and never slows. The player's commands,
:meth:`Runner.hop`, :meth:`Runner.jump` and :meth:`Runner.slide`, change the
lane, the height and the body's shape. No GL, no clock: the caller says how much
time has passed.
"""
import math

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner.events import Event

START_SPEED = 10.0
MAX_SPEED = 24.0
#: Metres per second gained each second, until MAX_SPEED.
ACCELERATION = 0.12
#: Seconds a hop takes to carry the body to the next path.
HOP_TIME = 0.16
GRAVITY = 42.0
JUMP_SPEED = math.sqrt(2.0 * GRAVITY * c.JUMP_PEAK)
JUMP_TIME = 2.0 * JUMP_SPEED / GRAVITY
#: Downward speed of a slide pressed in the air, which brings the runner to the
#: ground to slide there.
FAST_FALL_SPEED = 22.0
SLIDE_TIME = 0.8


class Runner:
    """The body that runs the course."""

    def __init__(self, course: c.Course, lane: int = 1) -> None:
        self.course = course
        #: The path the runner is on or hopping to.
        self.lane = lane
        #: How far the body is from the middle of its path, across the hill, while a hop
        #: carries it over.
        self.offset = 0.0
        self.distance = 0.0
        self.speed = START_SPEED
        #: Metres the feet are above the ground.
        self.height = 0.0
        self._rise = 0.0
        self._slide_left = 0.0
        self._slide_on_landing = False
        self._events: list[Event] = []
        #: Whether the body is knocked down: it does as it is told by nothing.
        self.stunned = False

    # -- state --
    @property
    def grounded(self) -> bool:
        """Whether the feet are on the ground."""
        return self.height <= 0.0 and self._rise <= 0.0

    @property
    def rising(self) -> bool:
        """Whether the body is on its way up in a jump."""
        return self._rise > 0.0

    @property
    def sliding(self) -> bool:
        """Whether the body is low in a slide."""
        return self._slide_left > 0.0

    @property
    def x(self) -> float:
        """Across the hill, where the body is."""
        return self.course.lane_x(self.lane, self.distance) + self.offset

    @property
    def band(self) -> tuple[float, float]:
        """Metres above the ground, bottom and top, that the body fills."""
        top = c.SLIDE_HEIGHT if self.sliding else c.STAND_HEIGHT
        return self.height, self.height + top

    def drain_events(self) -> list[Event]:
        """What the body did since this was last called."""
        events, self._events = self._events, []
        return events

    # -- commands --
    def hop(self, direction: int) -> bool:
        """Move one path left (-1) or right (+1); False if there is none, or a hedge is between
        them, or they are drawn too far apart."""
        if self.stunned:
            return False
        target = self.lane + direction
        if target not in self.course.lanes_at(self.distance):
            return False
        reach = self.distance + self.speed * HOP_TIME
        if self.course.blocks_hop(self.lane, target, self.distance, reach):
            self._events.append(Event.BLOCKED)
            return False
        self._move_to(target)
        self._events.append(Event.HOP)
        return True

    def knock_down(self) -> None:
        """Stop dead on the ground, whatever the body was doing, and stay down until
        :meth:`get_up`."""
        self.speed = 0.0
        self.height = 0.0
        self._rise = 0.0
        self._slide_left = 0.0
        self._slide_on_landing = False
        self.offset = 0.0
        self._events.clear()
        self.stunned = True

    def get_up(self) -> None:
        """Stand and run again from :data:`START_SPEED`: the speed gained is lost."""
        self.speed = START_SPEED
        self.stunned = False

    def _move_to(self, lane: int) -> None:
        """Change path without moving the body; the offset is what is left to glide."""
        here = self.x
        self.lane = lane
        self.offset = here - self.course.lane_x(lane, self.distance)

    def jump(self) -> bool:
        """Leave the ground; False in the air."""
        if self.stunned or not self.grounded:
            return False
        self._rise = JUMP_SPEED
        self._slide_left = 0.0
        self._slide_on_landing = False
        self._events.append(Event.JUMP)
        return True

    def slide(self) -> bool:
        """Drop low for a moment, or stay low for another moment if already sliding.

        In the air the runner falls at once and slides on landing. A runner who is knocked down
        does nothing.
        """
        if self.stunned:
            return False
        if self.sliding:
            self._slide_left = SLIDE_TIME
        elif self.grounded:
            self._slide_left = SLIDE_TIME
            self._events.append(Event.SLIDE)
        else:
            self._rise = -FAST_FALL_SPEED
            self._slide_on_landing = True
        return True

    # -- time --
    def advance(self, dt: float) -> None:
        """Run for ``dt`` seconds: faster, further down the hill, and through the air or a hop."""
        self.speed = min(MAX_SPEED, self.speed + ACCELERATION * dt)
        self.distance += self.speed * dt
        lanes = self.course.lanes_at(self.distance)
        if self.lane not in lanes:
            # The path has joined the trail: carry on in the nearest one.
            self._move_to(min(lanes, key=lambda lane: abs(lane - self.lane)))
        self._glide(dt)
        self._fly(dt)
        self._slide_left = max(0.0, self._slide_left - dt)

    def _glide(self, dt: float) -> None:
        """Carry the body onto the middle of its path, a path's width in HOP_TIME."""
        step = c.LANE_WIDTH / HOP_TIME * dt
        self.offset -= max(-step, min(step, self.offset))

    def _fly(self, dt: float) -> None:
        if self.grounded:
            return
        # Constant gravity integrates exactly, so the peak does not depend on the frame rate.
        self.height += self._rise * dt - 0.5 * GRAVITY * dt * dt
        self._rise -= GRAVITY * dt
        if self.height <= 0.0:
            self.height = 0.0
            self._rise = 0.0
            self._events.append(Event.LAND)
            if self._slide_on_landing:
                self._slide_on_landing = False
                self._slide_left = SLIDE_TIME
                self._events.append(Event.SLIDE)
