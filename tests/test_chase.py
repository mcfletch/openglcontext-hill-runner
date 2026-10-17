"""The point the camera looks at: the runner, a little ahead, with its sideways moves softened."""
import math

import numpy as np
import pytest

from openglcontext_hill_runner import chase
from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import runner as r


@pytest.fixture
def body():
    return r.Runner(c.Course(1))


def test_the_target_is_ahead_of_the_runner_and_above_the_ground(body):
    body.distance = 200.0
    target = chase.Chase(body).advance(1 / 60)
    assert -target[2] == pytest.approx(200.0 + chase.LOOK_AHEAD)
    assert target[1] == pytest.approx(c.ground_height(200.0 + chase.LOOK_AHEAD) + chase.HEIGHT)


def test_a_hop_moves_the_target_across_more_slowly_than_the_runner(body):
    follower = chase.Chase(body)
    start = follower.advance(1 / 60)[0]
    body.hop(+1)
    for _ in range(6):
        body.advance(1 / 60)
        target = follower.advance(1 / 60)
    assert start < target[0] < body.x


def test_the_target_settles_over_the_new_path(body):
    follower = chase.Chase(body)
    body.hop(-1)
    for _ in range(120):
        body.advance(1 / 60)
        target = follower.advance(1 / 60)
    assert target[0] == pytest.approx(
        body.course.lane_x(0, body.distance + chase.LOOK_AHEAD), abs=0.05)


def test_a_jump_lifts_the_target_by_part_of_the_height(body):
    follower = chase.Chase(body)
    flat = follower.advance(1 / 60)[1]
    body.jump()
    for _ in range(18):
        body.advance(1 / 60)
    lifted = follower.advance(1 / 60)[1]
    assert 0.0 < lifted - flat < body.height


def test_the_view_stays_over_the_trail_as_it_wanders():
    world = c.Course(5)
    body = r.Runner(world)
    follower = chase.Chase(body)
    offsets = []
    for _ in range(60 * 40):
        body.advance(1 / 60)
        target = follower.advance(1 / 60)
        offsets.append(target[0] - world.lane_x(1, body.distance + chase.LOOK_AHEAD))
    assert abs(world.centre(body.distance)) > 1.0
    assert np.abs(offsets).max() < 0.05


def test_the_view_follows_a_runner_along_a_path_that_draws_away():
    world = c.Course(3)
    fork = next(f for f in world.forks_between(0.0, 12000.0) if f.side == 1)
    body = r.Runner(world, lane=2)
    body.distance = fork.start + 1.0
    follower = chase.Chase(body)
    body.hop(+1)
    worst = 0.0
    for _ in range(60 * 8):
        body.advance(1 / 60)
        target = follower.advance(1 / 60)
        worst = max(worst, abs(target[0] - body.x))
    assert body.x - world.lane_x(2, body.distance) > 10.0
    assert worst < 1.5
    assert abs(target[0] - world.lane_x(3, body.distance + chase.LOOK_AHEAD)) < 0.4


def test_the_view_turns_to_where_the_runners_path_is_going():
    world = c.Course(5)
    body = r.Runner(world)
    follower = chase.Chase(body)
    body.distance = 700.0
    ahead = world.lane_x(1, 700.0 + chase.HEADING_AHEAD) - world.lane_x(1, 700.0)
    assert ahead != 0.0
    assert math.copysign(1.0, follower.heading()) == math.copysign(1.0, ahead)
    assert follower.heading() == pytest.approx(math.atan(ahead / chase.HEADING_AHEAD))


def test_the_view_turns_further_on_a_path_that_is_drawing_away():
    world = c.Course(3)
    fork = next(f for f in world.forks_between(0.0, 12000.0) if f.side == 1)
    body = r.Runner(world, lane=3)
    body.distance = fork.start + 60.0
    assert chase.Chase(body).heading() > 0.15


# -- under a ceiling --------------------------------------------------------------------

def camera_top(follower, body):
    """How high the camera is above the floor of the path at the runner, and the lowest the
    ceiling is from the camera to a few metres ahead of it."""
    target = follower.advance(1 / 60)
    up = follower.camera_offset()[1]
    camera_y = target[1] + up
    world = body.course
    along = body.distance + chase.LOOK_AHEAD - chase.CAMERA_BACK + np.linspace(0.0, chase.AHEAD_OF_CAMERA, 6)
    roof = min(float(c.ground_height(d)) + float(world.ceiling(d)) for d in along)
    return camera_y, roof


def low_and_high_places(world):
    cave = world.caves_between(0.0, 60000.0)[0]
    d = np.linspace(cave.start + 20.0, cave.end - 60.0, 400)
    roof = np.asarray(world.ceiling(d))
    return float(d[roof.argmin()]), float(d[roof.argmax()])


def test_in_the_open_the_camera_stands_where_it_always_has(body):
    body.distance = 200.0
    follower = chase.Chase(body)
    follower.advance(1 / 60)
    assert follower.camera_offset()[1] == pytest.approx(chase.CAMERA_UP)


def test_the_camera_stays_below_a_low_ceiling_and_above_the_runners_head():
    world = c.Course(1)
    low, _high = low_and_high_places(world)
    body = r.Runner(world)
    body.distance = low
    follower = chase.Chase(body)
    for _ in range(3):
        top, roof = camera_top(follower, body)
    assert top < roof - chase.CEILING_CLEARANCE + 0.05
    assert top > float(c.ground_height(low)) + c.STAND_HEIGHT * 0.9


def test_under_a_high_ceiling_the_camera_is_where_it_is_in_the_open():
    world = c.Course(1)
    _low, high = low_and_high_places(world)
    body = r.Runner(world)
    body.distance = high
    follower = chase.Chase(body)
    for _ in range(300):
        follower.advance(1 / 60)
    assert follower.camera_offset()[1] == pytest.approx(chase.CAMERA_UP)


def test_the_camera_is_never_through_the_ceiling_on_the_way_in_and_rises_gently_on_the_way_out():
    world = c.Course(1)
    cave = world.caves_between(0.0, 60000.0)[0]
    body = r.Runner(world)
    body.distance = cave.start - 30.0
    follower = chase.Chase(body)
    ups = []
    while body.distance < cave.end + 30.0:
        body.distance += 24.0 / 60
        top, roof = camera_top(follower, body)
        assert top < roof - chase.CEILING_CLEARANCE + 0.05 or follower.camera_offset()[1] <= chase.MIN_UP
        ups.append(follower.camera_offset()[1])
    rises = np.diff(ups)
    assert min(ups) < chase.CAMERA_UP and ups[-1] == pytest.approx(chase.CAMERA_UP, abs=0.05)
    assert rises.max() <= chase.RISE_RATE * chase.CAMERA_UP / 60 + 1e-9
