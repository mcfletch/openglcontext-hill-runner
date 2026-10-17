"""What the runner looks and sounds like, as a function of what it is doing. No GL.

The window asks :func:`pose_for` which clip to play each frame, hands
:func:`sounds_for` the events of the frame, and asks :class:`Footsteps` when a
foot lands. None of it touches GL, so tests run it without a window.
"""
from dataclasses import dataclass

from openglcontext_hill_runner import game as g
from openglcontext_hill_runner.events import Event

#: Seconds one cycle of each looping clip lasts at speed 1.0: two footfalls.
CYCLE = {'Jog_Fwd_Loop': 0.933, 'Sprint_Loop': 0.667}
#: The runner jogs below this speed and sprints from it, in metres a second.
SPRINT_FROM = 13.0
#: How much faster than its own pace each clip plays, for each metre a second of speed.
JOG_QUICKENING = 0.10
SPRINT_QUICKENING = 0.05
#: The clip lengths the jump and the slide are fitted to: the takeoff lasts 1.33
#: seconds as authored, and the airborne half of a jump about 0.3.
TAKEOFF_SPEED = 2.4

CLIPS = ('Jog_Fwd_Loop', 'Sprint_Loop', 'Jump_Start', 'Jump_Loop', 'Slide', 'Death01')

MUSIC = 'music'
STEP = 'footstep'


@dataclass(frozen=True)
class Pose:
    """A clip to play: how it ends, how long to blend into it, how fast it goes."""

    clip: str
    loop: bool
    fade: float
    speed: float = 1.0


def pose_for(game: g.Game) -> Pose:
    """The clip the runner is in this moment."""
    runner = game.runner
    if game.state is g.State.CRASHED:
        return Pose('Death01', False, 0.08, 1.6)
    if runner.sliding:
        return Pose('Slide', False, 0.08)
    if not runner.grounded:
        if runner.rising:
            return Pose('Jump_Start', False, 0.05, TAKEOFF_SPEED)
        return Pose('Jump_Loop', True, 0.1)
    if runner.speed < SPRINT_FROM:
        return Pose('Jog_Fwd_Loop', True, 0.15,
                    1.0 + JOG_QUICKENING * (runner.speed - 10.0))
    return Pose('Sprint_Loop', True, 0.15,
                1.0 + SPRINT_QUICKENING * (runner.speed - SPRINT_FROM))


@dataclass(frozen=True)
class Sound:
    """A one-shot sound: a file's name without its suffix, how loud, how fast."""

    name: str
    gain: float = 1.0
    rate: float = 1.0


_SOUNDS = {
    Event.JUMP: Sound('jump_grunt', 0.9),
    Event.SLIDE: Sound('slide', 0.8),
    Event.COIN: Sound('coin', 0.7),
    Event.JEWEL: Sound('chime', 0.8),
    Event.CRASH: Sound('crash', 1.0),
    Event.HIT: Sound('crash', 0.7, 1.15),
    Event.HEAL: Sound('chime', 0.5, 0.8),
    Event.HOP: Sound(STEP, 0.5),
    Event.LAND: Sound(STEP, 0.9),
}


def sounds_for(events: list[Event]) -> list[Sound]:
    """The sounds that go with ``events``; an event with none is silent."""
    return [_SOUNDS[event] for event in events if event in _SOUNDS]


class Footsteps:
    """Tells when a foot lands, at the pace of the stride the runner is animated at."""

    def __init__(self) -> None:
        self._phase = 0.0

    def advance(self, dt: float, game: g.Game) -> bool:
        """Move ``dt`` seconds on; True if a foot landed in that time."""
        runner = game.runner
        if game.state is not g.State.RUNNING or not runner.grounded or runner.sliding:
            self._phase = 0.0
            return False
        pose = pose_for(game)
        self._phase += dt * 2.0 * pose.speed / CYCLE[pose.clip]
        if self._phase >= 1.0:
            self._phase -= 1.0
            return True
        return False
