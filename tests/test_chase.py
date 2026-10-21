"""The point the camera looks at: the runner, a little ahead, with its sideways moves softened."""
import math

import numpy as np
import pytest

from openglcontext_hill_runner import chase
from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import landscape as ls
from openglcontext_hill_runner import runner as r
from openglcontext_hill_runner import scenery as s
from openglcontext_hill_runner import tunnel as tn


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


def test_the_camera_is_never_through_the_ceiling_and_glides_down_and_up():
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
    steps = np.diff(ups) * 60
    assert min(ups) < chase.CAMERA_UP and ups[-1] == pytest.approx(chase.CAMERA_UP, abs=0.05)
    assert np.abs(steps).max() < 5.0
    assert np.abs(np.diff(steps)).max() * 60 < 80.0


def test_where_the_camera_is_ducked_it_stands_to_the_right_and_looks_across_to_the_left():
    world = c.Course(1)
    low, _high = low_and_high_places(world)
    body = r.Runner(world)
    body.distance = low
    follower = chase.Chase(body)
    for _ in range(60):
        target = follower.advance(1 / 60)
    eye_x = target[0] + follower.camera_offset()[0]
    path_x = world.lane_x(body.lane, body.distance + chase.LOOK_AHEAD
                          - chase.CAMERA_BACK * math.cos(follower.heading()))
    assert eye_x - path_x >= 0.8 * chase.SIDE_STEP
    assert target[0] < body.x - 0.8 * chase.LOOK_ASIDE
    assert follower.sunk() > 0.8


def test_in_the_open_the_camera_stands_square_behind_the_runner_and_looks_straight_ahead(body):
    body.distance = 200.0
    follower = chase.Chase(body)
    target = follower.advance(1 / 60)
    sideways = follower.camera_offset()[0] + chase.CAMERA_BACK * math.sin(follower.heading())
    assert sideways == pytest.approx(0.0, abs=1e-9)
    assert target[0] == pytest.approx(body.x + (body.course.lane_x(body.lane, 203.0)
                                                - body.course.lane_x(body.lane, 200.0)))


def test_the_camera_comes_back_in_line_as_it_rises_out_of_the_ceiling():
    world = c.Course(1)
    _low, high = low_and_high_places(world)
    body = r.Runner(world)
    body.distance = high
    follower = chase.Chase(body)
    for _ in range(300):
        follower.advance(1 / 60)
    assert follower.camera_offset()[0] + chase.CAMERA_BACK * math.sin(follower.heading()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize('seed', [1, 2, 3, 4])
def test_the_camera_stays_clear_of_the_floor_of_the_tunnel(seed):
    land = ls.Landscape(c.Course(seed))
    world = land.course
    cave = world.caves_between(0.0, 60000.0)[0]
    floor = tn.bore(land, cave, lambda: s._fixings(land, cave)).floor_points
    lowest = np.inf
    for lane in world.lanes_at(cave.start + 10.0):
        body = r.Runner(world)
        body.lane = lane
        body.distance = cave.start - 40.0
        body.speed = 24.0
        follower = chase.Chase(body, land)
        while body.distance < cave.end + 20.0:
            body.distance += 24.0 / 60
            target = follower.advance(1 / 60)
            eye = np.array(target) + np.array(follower.camera_offset()) * (1.0 + chase.PULL_BACK_MAX)
            if cave.start + 2.0 < -eye[2] < cave.end - 2.0:
                near = np.hypot(floor[:, 0] - eye[0], floor[:, 2] - eye[2]).argmin()
                lowest = min(lowest, eye[1] - floor[near, 1])
    assert lowest >= 0.6


# -- the floor under the camera is looked up for a stretch of the run at a time -------------

@pytest.fixture
def lookups(monkeypatch):
    """A landscape on the first course, and a list of the points its ground was asked about."""
    land = ls.Landscape(c.Course(1))
    asked = []
    original = land.footing
    monkeypatch.setattr(land, 'footing', lambda x, d: asked.append(np.size(d)) or original(x, d))
    return land, asked


def test_the_floor_under_the_camera_is_looked_up_once_for_the_next_few_metres(lookups):
    land, asked = lookups
    body = r.Runner(land.course)
    body.distance = 500.0
    follower = chase.Chase(body, land)
    follower.advance(1 / 60)
    assert len(asked) == 1
    for _ in range(int(0.9 * chase.FLOOR_STALE / 0.1)):
        body.distance += 0.1
        follower.advance(1 / 60)
    assert len(asked) == 1


def test_the_floor_is_looked_up_again_once_the_run_has_gone_on(lookups):
    land, asked = lookups
    body = r.Runner(land.course)
    body.distance = 500.0
    follower = chase.Chase(body, land)
    follower.advance(1 / 60)
    body.distance += chase.FLOOR_STALE + 0.1
    follower.advance(1 / 60)
    assert len(asked) == 2


def test_the_floor_is_looked_up_again_on_another_path(lookups):
    land, asked = lookups
    body = r.Runner(land.course)
    body.distance = 500.0
    follower = chase.Chase(body, land)
    follower.advance(1 / 60)
    body.lane = 0
    follower.advance(1 / 60)
    assert len(asked) == 2


def test_the_floor_is_looked_up_again_when_the_run_goes_back(lookups):
    land, asked = lookups
    body = r.Runner(land.course)
    body.distance = 500.0
    follower = chase.Chase(body, land)
    follower.advance(1 / 60)
    body.distance -= 1.0
    follower.advance(1 / 60)
    assert len(asked) == 2


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_a_floor_looked_up_a_few_metres_back_is_never_lower_than_one_looked_up_here(seed):
    """What the camera is kept above is the highest ground under it; the stretch it is looked up
    over reaches as far as the run goes before it is looked up again."""
    land = ls.Landscape(c.Course(seed))
    body = r.Runner(land.course)
    held = chase.Chase(body, land)
    worst = np.inf
    for step in range(400):
        body.distance = 300.0 + step * 0.4
        held.advance(1 / 60)
        fresh = chase.Chase(body, land)
        worst = min(worst, held._floor_under_camera() - fresh._floor_under_camera())
    assert worst >= -0.05
