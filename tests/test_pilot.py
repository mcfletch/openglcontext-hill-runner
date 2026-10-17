"""The autopilot plays the game through the controls a player has.

It is the proof that every course can be run: a generator that made an
unavoidable obstacle would stop the pilot, and the test would name the seed.
"""
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner import pilot as p

FRAME = 1 / 60


def drive(seed, metres, branch=None):
    game = g.Game(c.Course(seed))
    pilot = p.Pilot(game, branch=branch)
    lanes_run = set()
    while game.runner.distance < metres and game.state is g.State.RUNNING:
        pilot.drive()
        game.advance(FRAME)
        lanes_run.add(game.runner.lane)
    game.lanes_run = lanes_run
    return game


@pytest.mark.parametrize('branch', [None, True, False])
@pytest.mark.parametrize('seed', range(20))
def test_the_pilot_runs_every_course_without_a_crash(seed, branch):
    game = drive(seed, 3500, branch)
    assert game.state is g.State.RUNNING, (
        'crashed into %s at %.1f m in lane %d' % (
            game.crashed_into, game.runner.distance, game.runner.lane))


def test_a_pilot_that_takes_the_forks_runs_along_the_new_paths():
    game = drive(3, 3500, branch=True)
    assert game.lanes_run & {3, 4, -1, -2}


def test_a_pilot_that_declines_the_forks_stays_on_the_trail():
    game = drive(3, 3500, branch=False)
    assert game.lanes_run <= {0, 1, 2}


def test_the_pilot_collects_treasure_on_its_way():
    game = drive(3, 1500)
    assert game.coins > 20


def test_the_pilot_hops_for_walls_and_jumps_and_slides():
    from openglcontext_hill_runner.events import Event
    game = g.Game(c.Course(5))
    pilot = p.Pilot(game)
    heard = set()
    while game.runner.distance < 2500 and game.state is g.State.RUNNING:
        pilot.drive()
        heard.update(game.advance(FRAME))
    assert {Event.HOP, Event.JUMP, Event.SLIDE} <= heard
