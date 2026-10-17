"""The runner's body: speed, lanes, jumping and sliding."""
import math

import numpy as np
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import runner as r
from openglcontext_hill_runner.events import Event

FRAME = 1 / 60


def run_for(body, seconds):
    for _ in range(round(seconds / FRAME)):
        body.advance(FRAME)


@pytest.fixture
def body():
    return r.Runner(c.Course(1))


def test_the_runner_starts_on_the_middle_path_at_the_line(body):
    assert (body.lane, body.distance, body.height) == (1, 0.0, 0.0)
    assert body.x == pytest.approx(body.course.lane_x(1, 0.0))
    assert body.speed == r.START_SPEED


def test_the_runner_covers_speed_times_time(body):
    body.advance(0.5)
    assert body.distance == pytest.approx(body.speed * 0.5, rel=0.02)


def test_the_runner_only_ever_speeds_up(body):
    speeds = []
    for _ in range(600):
        body.advance(FRAME)
        speeds.append(body.speed)
    assert all(b >= a for a, b in zip(speeds, speeds[1:], strict=False))
    assert speeds[-1] > speeds[0]


def test_the_runner_tops_out_at_full_speed(body):
    run_for(body, 400)
    assert body.speed == r.MAX_SPEED


def test_a_hop_changes_the_lane_at_once_and_the_body_follows(body):
    before = body.x
    assert body.hop(+1)
    assert body.lane == 2
    assert body.x == pytest.approx(before)
    run_for(body, r.HOP_TIME * 1.1)
    assert body.x == pytest.approx(body.course.lane_x(2, body.distance))


def test_the_runner_cannot_hop_off_the_hill(body):
    assert body.hop(+1)
    assert not body.hop(+1)
    assert body.lane == 2
    assert body.hop(-1) and body.hop(-1)
    assert not body.hop(-1)
    assert body.lane == 0


def test_hops_can_be_chained_before_the_first_finishes(body):
    assert body.hop(+1)
    run_for(body, 0.05)
    assert body.hop(-1) and body.hop(-1)
    assert body.lane == 0


def test_a_hedge_refuses_the_hop_and_says_so():
    world = c.Course(11)
    divider = next(d for i in range(120) for d in world.chunk(i).dividers)
    body = r.Runner(world, lane=divider.lane)
    body.distance = (divider.start + divider.end) / 2
    assert not body.hop(+1)
    assert body.lane == divider.lane
    assert body.drain_events() == [Event.BLOCKED]


def test_a_jump_rises_to_the_peak_and_comes_down(body):
    assert body.jump()
    peak = 0.0
    for _ in range(round(1.0 / FRAME)):
        body.advance(FRAME)
        peak = max(peak, body.height)
    assert peak == pytest.approx(c.JUMP_PEAK, rel=0.05)
    assert body.height == 0.0
    assert body.grounded


def test_a_jump_lasts_the_time_gravity_gives_it(body):
    body.jump()
    airborne = 0
    while not body.grounded or airborne == 0:
        body.advance(FRAME)
        airborne += 1
        assert airborne < 200
    assert airborne * FRAME == pytest.approx(r.JUMP_TIME, abs=0.05)


def test_there_is_no_second_jump_in_the_air(body):
    assert body.jump()
    run_for(body, 0.1)
    assert not body.jump()


def test_a_slide_lowers_the_body_for_its_duration(body):
    assert body.slide()
    assert body.band == (0.0, c.SLIDE_HEIGHT)
    run_for(body, r.SLIDE_TIME + 0.05)
    assert body.band == (0.0, c.STAND_HEIGHT)
    assert not body.sliding


def test_sliding_again_restarts_the_slide(body):
    body.slide()
    run_for(body, r.SLIDE_TIME * 0.75)
    assert body.slide()
    run_for(body, r.SLIDE_TIME * 0.5)
    assert body.sliding
    assert body.drain_events() == [Event.SLIDE]


def test_standing_the_body_fills_its_full_height(body):
    assert body.band == (0.0, c.STAND_HEIGHT)


def test_a_jump_cuts_a_slide_short(body):
    body.slide()
    assert body.jump()
    assert not body.sliding


def test_a_slide_in_the_air_drops_the_runner_and_slides_on_landing(body):
    body.jump()
    run_for(body, 0.15)
    assert body.slide()
    run_for(body, 0.3)
    assert body.grounded
    assert body.sliding


def test_events_report_what_the_body_did_since_they_were_last_read(body):
    body.jump()
    run_for(body, 1.0)
    body.slide()
    body.hop(-1)
    assert body.drain_events() == [Event.JUMP, Event.LAND, Event.SLIDE, Event.HOP]
    assert body.drain_events() == []


def test_the_body_is_rising_only_on_the_way_up(body):
    assert not body.rising
    body.jump()
    body.advance(0.05)
    assert body.rising
    run_for(body, 0.3)
    assert body.height > 0 and not body.rising


def test_the_jump_speed_follows_from_the_peak_and_gravity():
    assert r.JUMP_SPEED == pytest.approx(math.sqrt(2 * r.GRAVITY * c.JUMP_PEAK))


# -- following the paths of a fork -------------------------------------------------

def fork_course():
    world = c.Course(3)
    fork = next(f for f in world.forks_between(0.0, 12000.0) if f.side == 1)
    return world, fork


def test_a_runner_cannot_hop_to_a_path_that_is_not_there(body):
    body.hop(+1)
    assert not body.hop(+1)
    assert body.drain_events() == [Event.HOP]


def test_a_runner_follows_its_path_as_it_draws_away_from_the_others():
    world, fork = fork_course()
    body = r.Runner(world, lane=2)
    body.distance = fork.start + 1.0
    assert body.hop(+1) and body.lane == 3
    run_for(body, 6.0)
    assert body.distance > fork.start + 60.0
    assert body.x - world.lane_x(2, body.distance) > 3.0
    assert body.x == pytest.approx(world.lane_x(3, body.distance), abs=0.02)


def test_a_hop_does_not_move_the_runner_across_at_once_on_a_path_off_centre():
    world, fork = fork_course()
    body = r.Runner(world, lane=2)
    body.distance = fork.start + 2.0
    before = body.x
    body.hop(+1)
    assert body.x == pytest.approx(before)
    assert body.offset == pytest.approx(before - world.lane_x(3, body.distance))


def test_the_paths_drawn_apart_cannot_be_hopped_between_again():
    world, fork = fork_course()
    body = r.Runner(world, lane=3)
    body.distance = (fork.start + fork.end) / 2
    assert not body.hop(-1)
    assert body.drain_events() == [Event.BLOCKED]


def test_a_runner_on_a_path_that_joins_the_trail_is_carried_onto_the_nearest_one():
    world, fork = fork_course()
    body = r.Runner(world, lane=4)
    body.distance = fork.end - 3.0
    body.speed = 15.0
    positions = []
    for _ in range(90):
        body.advance(FRAME)
        positions.append(body.x)
    assert body.lane == 2
    assert np.abs(np.diff(positions)).max() < 0.2
    assert body.x == pytest.approx(world.lane_x(2, body.distance), abs=0.02)
