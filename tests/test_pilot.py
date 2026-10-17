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
    assert game.damage == 0, (
        'crashed into %s at %.1f m in lane %d' % (
            game.crashed_into, game.runner.distance, game.runner.lane))


@pytest.mark.parametrize('seed', range(4))
def test_the_pilot_runs_the_hardest_stretch_too(seed):
    start = c.HARDEST_AT_CHUNK * c.CHUNK_LENGTH
    game = drive(seed, start + 800.0)
    assert game.damage == 0, 'crashed into %s at %.1f m' % (
        game.crashed_into, game.runner.distance)


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


@pytest.mark.parametrize('branch', [None, True, False])
@pytest.mark.parametrize('seed', [101, 202, 303, 404])
def test_the_pilot_runs_on_for_thirty_kilometres_without_a_crash(seed, branch):
    """The menu's Demo hands the run to the pilot to play for as long as it is left."""
    game = drive(seed, 30_000, branch)
    assert game.damage == 0, 'crashed into %s at %.1f m' % (
        game.crashed_into, game.runner.distance)


def clear_start(world, start):
    """``start``, or the first place before it with no obstacle within a few metres."""
    while world.obstacles_between(start - 6.0, start + 30.0):
        start -= 10.0
    return start


@pytest.mark.parametrize('speed', [10.0, 24.0])
@pytest.mark.parametrize('seed', range(6))
def test_the_pilot_runs_through_every_cave_without_a_hit_at_either_speed(seed, speed):
    world = c.Course(seed)
    for cave in world.caves_between(0.0, 30000.0)[:8]:
        game = g.Game(world)
        pilot = p.Pilot(game)
        game.runner.distance = clear_start(world, cave.start - 250.0)
        game.runner.speed = speed
        while game.runner.distance < cave.end + 30.0 and game.state is g.State.RUNNING:
            game.runner.speed = max(game.runner.speed, speed)
            pilot.drive()
            game.advance(FRAME)
        assert game.damage == 0, 'hit %s at %.1f m in cave %d of seed %d, going %g m/s' % (
            game.crashed_into, game.runner.distance, cave.index, seed, speed)
        assert game.runner.distance >= cave.end + 30.0


def test_the_pilot_never_jumps_under_a_low_ceiling():
    world = c.Course(2)
    game = g.Game(world)
    pilot = p.Pilot(game)
    cave = world.caves_between(0.0, 30000.0)[0]
    game.runner.distance = clear_start(world, cave.start - 100.0)
    jumped_under = 0
    while game.runner.distance < cave.end and game.state is g.State.RUNNING:
        pilot.drive()
        game.advance(FRAME)
        if not game.runner.grounded and float(world.ceiling(game.runner.distance)) < c.JUMP_NEEDS:
            jumped_under += 1
    assert jumped_under == 0
