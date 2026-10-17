"""A session: one run, advanced a frame at a time, as the window will advance it."""

import math

import pytest

from openglcontext_hill_runner import chase, cues
from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner import landscape as ls
from openglcontext_hill_runner import runner as r
from openglcontext_hill_runner import session as sess
from openglcontext_hill_runner.session import GAME_OVER_DELAY, HOP_LEAN, MAX_FRAME, SLOPE_LEAN

FRAME = 1 / 60


def Session(course, demo=False):
    return sess.Session(ls.Landscape(course), demo=demo)


def test_a_frame_reports_the_pose_the_sounds_and_where_to_look():
    session = Session(c.Course(1))
    frame = session.step(FRAME)
    assert frame.pose == cues.pose_for(session.game)
    assert frame.target[2] < 0
    assert frame.sounds == []


def test_a_command_given_between_frames_is_heard_in_the_next():
    session = Session(c.Course(1))
    session.game.runner.jump()
    frame = session.step(FRAME)
    assert [s.name for s in frame.sounds] == ['jump_grunt']
    assert frame.pose.clip == 'Jump_Start'


def test_footfalls_are_heard_as_the_runner_runs():
    session = Session(c.Course(1))
    steps = sum(cues.STEP in [s.name for s in session.step(FRAME).sounds] for _ in range(120))
    assert 4 <= steps <= 9


def test_the_run_is_over_a_moment_after_the_crash():
    world = c.Course(1)
    session = Session(world)
    session.game.runner.distance = 0.0
    walls = tuple(c.Obstacle(c.Kind.WALL, 1, d) for d in (12.0, 50.0, 90.0))
    world.chunk(0)
    world._chunks[0] = c.Chunk(0, 0.0, walls, (), ())
    heard = []
    for _ in range(60 * 30):
        heard += [s.name for s in session.step(FRAME).sounds]
        if session.game.state is g.State.CRASHED:
            break
    assert session.game.damage == g.MAX_DAMAGE
    assert heard.count('crash') == g.MAX_DAMAGE
    assert not session.over
    for _ in range(round(GAME_OVER_DELAY / FRAME) + 2):
        session.step(FRAME)
    assert session.over


def test_the_demo_pilot_runs_a_course_unaided():
    session = Session(c.Course(2), demo=True)
    for _ in range(60 * 30):
        session.step(FRAME)
    assert session.game.damage == 0
    assert session.game.runner.distance > 200
    assert session.game.coins > 0


def test_without_a_pilot_the_runner_that_does_nothing_eventually_crashes():
    session = Session(c.Course(2))
    for _ in range(60 * 240):
        session.step(FRAME)
        if session.game.state is g.State.CRASHED:
            break
    assert session.game.state is g.State.CRASHED


# -- where the runner stands, and where the camera is, each frame ---------------------

def test_the_runner_stands_on_the_land_and_rises_with_a_jump():
    session = Session(c.Course(1))
    land = session.land
    flat = session.step(FRAME)
    runner = session.game.runner
    assert flat.position == pytest.approx(
        (runner.x, float(land.height(runner.x, runner.distance)), -runner.distance))
    session.game.runner.jump()
    for _ in range(20):
        up = session.step(FRAME)
    assert up.position[1] == pytest.approx(
        float(land.height(runner.x, runner.distance)) + runner.height)
    assert runner.height > 0.3


def test_the_runner_faces_where_its_path_goes_and_is_tipped_down_the_slope():
    session = Session(c.Course(5))
    session.game.runner.distance = 700.0
    frame = session.step(FRAME)
    assert frame.heading == pytest.approx(session.chase.heading())
    d = session.game.runner.distance
    fall = (c.ground_height(d - 1.0) - c.ground_height(d + 1.0)) / 2.0
    assert frame.pitch == pytest.approx(-math.atan(fall) * SLOPE_LEAN)
    assert frame.pitch < 0.0


def test_the_runner_leans_into_a_hop_and_stands_straight_otherwise():
    session = Session(c.Course(1))
    for _ in range(5):
        assert session.step(FRAME).lean == pytest.approx(0.0, abs=1e-9)
    session.game.runner.hop(+1)
    leans = [session.step(FRAME).lean for _ in range(6)]
    assert min(leans) < -0.5 * HOP_LEAN and max(leans) <= 0.0
    for _ in range(30):
        frame = session.step(FRAME)
    assert frame.lean == pytest.approx(0.0, abs=1e-6)
    session.game.runner.hop(-1)
    assert max(session.step(FRAME).lean for _ in range(6)) > 0.5 * HOP_LEAN


def test_the_camera_stands_behind_the_runner_along_its_path():
    session = Session(c.Course(5))
    session.game.runner.distance = 700.0
    frame = session.step(FRAME)
    turn = session.chase.heading()
    assert frame.camera_offset == pytest.approx((-chase.CAMERA_BACK * math.sin(turn),
                                                  chase.CAMERA_UP,
                                                  chase.CAMERA_BACK * math.cos(turn)))
    assert math.hypot(frame.camera_offset[0], frame.camera_offset[2]) == pytest.approx(
        chase.CAMERA_BACK)


def test_a_long_gap_between_frames_is_run_as_a_short_one():
    session = Session(c.Course(1))
    frame = session.step(5.0)
    assert frame.dt == MAX_FRAME
    assert session.game.runner.distance == pytest.approx(r.START_SPEED * MAX_FRAME, rel=0.05)


def test_the_pilot_can_be_handed_the_run_and_give_it_back():
    session = Session(c.Course(2))
    assert session.demo is False and session.pilot is None
    session.demo = True
    assert session.demo is True and session.pilot is not None
    pilot = session.pilot
    session.demo = True
    assert session.pilot is pilot                       # asking again does not start another
    for _ in range(600):
        session.step(1 / 60)
    assert session.game.state.name == 'RUNNING' and session.game.runner.distance > 50.0
    session.demo = False
    assert session.demo is False and session.pilot is None


def test_a_run_the_pilot_took_over_part_way_goes_on_without_a_crash():
    session = Session(c.Course(4))
    for _ in range(300):
        session.step(1 / 60)
    session.demo = True
    while session.game.runner.distance < 3000.0 and session.game.state.name == 'RUNNING':
        session.step(1 / 60)
    assert session.game.state.name == 'RUNNING'


def test_the_pilot_does_nothing_while_the_runner_is_down():
    world = c.Course(1)
    session = Session(world, demo=True)
    session.game.state = g.State.DOWN
    session.game._down_for = 5.0
    session.game.runner.stunned = True
    wall = c.Obstacle(c.Kind.WALL, 1, 12.0)
    world.chunk(0)
    world._chunks[0] = c.Chunk(0, 0.0, (wall,), (), ())
    for _ in range(30):
        session.step(FRAME)
    assert session.game.runner.grounded and session.game.runner.lane == 1
    assert session.game.runner.drain_events() == []


def test_a_hit_runner_stays_where_it_fell_in_the_picture_until_it_is_up():
    world = c.Course(1)
    session = Session(world)
    wall = c.Obstacle(c.Kind.WALL, 1, 12.0)
    world.chunk(0)
    world._chunks[0] = c.Chunk(0, 0.0, (wall,), (), ())
    while session.game.state is not g.State.DOWN:
        session.step(FRAME)
    where = session.step(FRAME).position
    for _ in range(int(g.STUMBLE_TIME * 60) - 6):
        assert session.step(FRAME).position == where
    while session.game.state is g.State.DOWN:
        session.step(FRAME)
    assert session.step(1 / 30).position != where


def test_a_runner_over_a_crevice_stands_on_the_lip_and_does_not_follow_the_floor_down():
    world = c.Course(1)
    session = Session(world)
    crevice = next(o for cave in world.caves_between(0.0, 30000.0)
                   for o in world.obstacles_between(cave.start, cave.end)
                   if o.kind is c.Kind.CREVICE)
    session.game.runner.distance = crevice.distance
    session.game.runner._rise = 1.0
    session.game.runner.height = 1.0
    frame = session.step(FRAME)
    runner = session.game.runner
    assert frame.position[1] == pytest.approx(
        float(session.land.footing(runner.x, runner.distance)) + runner.height)
    assert frame.position[1] > float(session.land.height(runner.x, runner.distance)) + runner.height + 2.0
