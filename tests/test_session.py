"""A session: one run, advanced a frame at a time, as the window will advance it."""

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import cues
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner.session import GAME_OVER_DELAY, Session

FRAME = 1 / 60


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
    wall = c.Obstacle(c.Kind.WALL, 1, 12.0)
    world.chunk(0)
    world._chunks[0] = c.Chunk(0, 0.0, (wall,), (), ())
    heard = []
    for _ in range(300):
        heard += [s.name for s in session.step(FRAME).sounds]
        if session.game.state is g.State.CRASHED:
            break
    assert 'crash' in heard
    assert not session.over
    for _ in range(round(GAME_OVER_DELAY / FRAME) + 2):
        session.step(FRAME)
    assert session.over


def test_the_demo_pilot_runs_a_course_unaided():
    session = Session(c.Course(2), demo=True)
    for _ in range(60 * 30):
        session.step(FRAME)
    assert session.game.state is g.State.RUNNING
    assert session.game.runner.distance > 200
    assert session.game.coins > 0


def test_without_a_pilot_the_runner_that_does_nothing_eventually_crashes():
    session = Session(c.Course(2))
    for _ in range(60 * 60):
        session.step(FRAME)
        if session.game.state is g.State.CRASHED:
            break
    assert session.game.state is g.State.CRASHED
