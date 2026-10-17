"""A run: the runner on the course, what it collects and what ends it. No GL."""
import enum
import math

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner.events import Event
from openglcontext_hill_runner.runner import Runner

#: The longest stretch of time simulated in one piece. At full speed the runner
#: covers 24 m a second, and an obstacle is under a metre deep.
MAX_STEP = 1.0 / 120.0
#: Half the width of the runner's body, and of its depth along the track.
BODY_HALF_WIDTH = 0.15
BODY_HALF_DEPTH = 0.25
#: How far from the middle of its path the runner can reach a coin, and how far
#: along the track, and how much of the coin's height counts.
COIN_REACH = 0.3
COIN_DEPTH = 0.6
#: Forgiveness at the feet: a toe over the top of a log is a clear jump.
FEET_FORGIVENESS = 0.08


class State(enum.Enum):
    """Whether the run is going or has ended."""

    RUNNING = 'running'
    CRASHED = 'crashed'


class Game:
    """One run down the hill."""

    def __init__(self, course: c.Course) -> None:
        self.course = course
        self.restart()

    def restart(self) -> None:
        """Back to the start line with nothing collected."""
        self.runner = Runner(self.course)
        self.state = State.RUNNING
        self.coins = 0
        self.points = 0
        self.crashed_into: c.Obstacle | None = None
        self._taken: set[tuple[int, int]] = set()

    @property
    def meters(self) -> int:
        """Whole metres run."""
        return int(self.runner.distance)

    def taken(self, coin: c.Coin) -> bool:
        """Whether the runner has taken ``coin``."""
        return coin.ident in self._taken

    def advance(self, dt: float) -> list[Event]:
        """Run for ``dt`` seconds; returns what happened, in order."""
        heard: list[Event] = []
        if self.state is not State.RUNNING:
            return heard
        steps = max(1, math.ceil(dt / MAX_STEP))
        for _ in range(steps):
            self.runner.advance(dt / steps)
            heard += self.runner.drain_events()
            heard += self._collect()
            if self._crashed():
                heard.append(Event.CRASH)
                break
        return heard

    # -- rules --
    def _crashed(self) -> bool:
        runner = self.runner
        low, high = runner.band
        reach = BODY_HALF_DEPTH + max(c.DEPTH.values()) / 2.0
        for obstacle in self.course.obstacles_between(runner.distance - reach,
                                                      runner.distance + reach):
            if abs(obstacle.distance - runner.distance) >= BODY_HALF_DEPTH + obstacle.depth / 2:
                continue
            if all(abs(runner.x - self.course.lane_x(lane, runner.distance))
                   >= c.OBSTACLE_HALF_WIDTH + BODY_HALF_WIDTH for lane in obstacle.lanes):
                continue
            bottom, top = obstacle.kind.band
            if low + FEET_FORGIVENESS < top and high > bottom:
                self.state = State.CRASHED
                self.crashed_into = obstacle
                return True
        return False

    def _collect(self) -> list[Event]:
        runner = self.runner
        low, high = runner.band
        heard = []
        for coin in self.course.coins_between(runner.distance - COIN_DEPTH,
                                              runner.distance + COIN_DEPTH):
            if coin.ident in self._taken:
                continue
            if abs(runner.x - self.course.lane_x(coin.lane, runner.distance)) >= COIN_REACH:
                continue
            if low < coin.height + c.COIN_HALF_HEIGHT and high > coin.height - c.COIN_HALF_HEIGHT:
                self._taken.add(coin.ident)
                self.coins += 1
                self.points += coin.value
                heard.append(Event.JEWEL if coin.value > 1 else Event.COIN)
        return heard
