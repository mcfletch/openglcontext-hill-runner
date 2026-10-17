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
#: Metres after a hit in which nothing hits the runner again, so one row of obstacles is one hit.
RECOVERY_DISTANCE = 15.0
#: How far a head may be into a ceiling before it counts, in metres.
HEAD_FORGIVENESS = 0.05
#: Seconds the runner is down after a hit before it runs on: the length of its roll to its feet.
STUMBLE_TIME = 1.3
#: What a continue costs, in coins, and how fast they are counted out: the whole of it in the time
#: the runner takes to get to its feet. A runner with fewer coins than that cannot buy one, keeps
#: what it has, and the run is over.
CONTINUE_COST = 100
PAY_RATE = CONTINUE_COST / STUMBLE_TIME
#: How far short of the obstacle it hit the runner is left standing, in metres.
KNOCKED_BACK = 0.1


class State(enum.Enum):
    """Whether the run is going or has ended."""

    RUNNING = 'running'
    #: Hit, and getting up while the continue is paid for: nothing moves, and nothing can hit the
    #: runner again.
    DOWN = 'down'
    #: A hit that the coins could not pay for.
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
        #: Coins in hand: a pickup adds its value, and a hit takes :data:`CONTINUE_COST`.
        self.coins = 0
        self.points = 0
        #: Hits taken, and coins of the continue for the latest still to be paid.
        self.hits = 0
        self.owed = 0
        self._due = 0.0
        #: The obstacle of the latest hit, or None for a head against a ceiling.
        self.crashed_into: c.Obstacle | None = None
        self._last_hit = 0.0
        self._down_for = 0.0
        self._taken: set[tuple[int, int]] = set()

    @property
    def recovering(self) -> bool:
        """Whether a hit was taken so recently that nothing can hit the runner again."""
        return self.hits > 0 and self.runner.distance < self._last_hit + RECOVERY_DISTANCE

    @property
    def paying(self) -> bool:
        """Whether a continue is being paid for: coins are counting down."""
        return self.owed > 0

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
        if self.state is State.DOWN:
            return self._get_up(dt)
        if self.state is not State.RUNNING:
            return heard
        steps = max(1, math.ceil(dt / MAX_STEP))
        for _ in range(steps):
            self.runner.advance(dt / steps)
            heard += self.runner.drain_events()
            heard += self._collect()
            if self._hit():
                self.hits += 1
                self._knock_back()
                self.runner.knock_down()
                if self.coins < CONTINUE_COST:
                    self.state = State.CRASHED
                    heard.append(Event.CRASH)
                else:
                    self.state = State.DOWN
                    self._down_for = STUMBLE_TIME
                    self.owed = CONTINUE_COST
                    self._due = 0.0
                    heard.append(Event.HIT)
                break
        return heard

    # -- rules --
    def _get_up(self, dt: float) -> list[Event]:
        """A step of the time down: count out the coins of the continue, and get up when it is paid
        for and the runner has rolled to its feet."""
        heard: list[Event] = []
        if self.owed > 0:
            self._due += PAY_RATE * dt
            taken = min(int(self._due), self.owed)
            self._due -= taken
            self.coins -= taken
            self.owed -= taken
            if self.owed == 0:
                heard.append(Event.PAID)
        self._down_for -= dt
        if self._down_for <= 0.0 and self.owed == 0:
            self.state = State.RUNNING
            self.runner.get_up()
            heard.append(Event.RISE)
        return heard

    def _knock_back(self) -> None:
        """Leave the runner standing just short of the obstacle it hit; the recovery and the
        run that heals are counted from there."""
        hit = self.crashed_into
        if hit is None:
            return
        short = hit.distance - hit.depth / 2.0 - BODY_HALF_DEPTH - KNOCKED_BACK
        self.runner.distance = min(self.runner.distance, short)
        self._last_hit = self.runner.distance

    def _hit(self) -> bool:
        """Whether the runner is in an obstacle it cannot be in, or its head is in the ceiling; records
        it if so."""
        if self.recovering:
            return False
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
            if obstacle.kind is c.Kind.CREVICE:
                # Only a runner clear of the floor crosses a crevice.
                fell = low < c.CREVICE_CLEAR
            else:
                fell = low + FEET_FORGIVENESS < top and high > bottom
            if fell:
                self.crashed_into = obstacle
                self._last_hit = runner.distance
                return True
        if high > float(self.course.ceiling(runner.distance)) - HEAD_FORGIVENESS:
            self.crashed_into = None
            self._last_hit = runner.distance
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
                self.coins += coin.value
                self.points += coin.value
                heard.append(Event.JEWEL if coin.value > 1 else Event.COIN)
        return heard
