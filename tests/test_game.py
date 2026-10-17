"""The rules: collisions, treasure, and the end of a run."""
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner.events import Event
from openglcontext_hill_runner.runner import START_SPEED

FRAME = 1 / 60


class Empty(c.Course):
    """A hill with nothing on it but what a test puts there."""

    def __init__(self, obstacles=(), coins=()):
        super().__init__(0)
        self.placed = (list(obstacles), list(coins))

    def _make(self, index):
        start = index * c.CHUNK_LENGTH
        obstacles = tuple(o for o in self.placed[0] if start <= o.distance < start + c.CHUNK_LENGTH)
        coins = tuple(k for k in self.placed[1] if start <= k.distance < start + c.CHUNK_LENGTH)
        return c.Chunk(index, start, obstacles, coins, ())


def play(game, until, command=None):
    """Advance in frames until ``until(game)``; returns the events heard."""
    heard = []
    for frame in range(60 * 240):
        if command is not None:
            command(game, frame)
        heard += game.advance(FRAME)
        if until(game):
            return heard
    pytest.fail('the condition never came')


def reaches(distance):
    return lambda game: game.runner.distance >= distance or game.state is g.State.CRASHED


def test_a_clear_hill_can_be_run_for_ever():
    game = g.Game(Empty())
    play(game, reaches(300))
    assert game.state is g.State.RUNNING
    assert game.meters >= 300


def test_running_into_a_wall_is_a_hit_and_the_run_goes_on():
    game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 1, 30.0)]))
    heard = play(game, reaches(60))
    assert game.state is g.State.RUNNING
    assert game.damage == 1
    assert Event.HIT in heard and Event.CRASH not in heard
    assert game.crashed_into.kind is c.Kind.WALL


def test_a_hop_clears_the_wall():
    game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 1, 30.0)]))
    play(game, reaches(60), lambda game, frame: frame == 5 and game.runner.hop(+1))
    assert game.state is g.State.RUNNING


def test_a_low_obstacle_is_jumped_over_and_stops_a_runner_who_stays_down():
    low = c.Obstacle(c.Kind.LOW, 1, 40.0)
    ran_into = g.Game(Empty([low]))
    play(ran_into, reaches(80))
    assert ran_into.damage == 1

    jumper = g.Game(Empty([low]))
    play(jumper, reaches(80), jump_before(jumper, 40.0))
    assert jumper.damage == 0


def jump_before(game, obstacle_distance):
    """A command that jumps when the obstacle is half a jump away."""
    def command(game, frame):
        gap = obstacle_distance - game.runner.distance
        if gap <= game.runner.speed * 0.3 and game.runner.grounded:
            game.runner.jump()
    return command


def test_a_high_obstacle_is_slid_under_and_hits_a_runner_who_stands():
    high = c.Obstacle(c.Kind.HIGH, 1, 40.0)
    standing = g.Game(Empty([high]))
    play(standing, reaches(80))
    assert standing.damage == 1

    sliding = g.Game(Empty([high]))

    def slide_before(game, frame):
        if 40.0 - game.runner.distance <= game.runner.speed * 0.2:
            game.runner.slide()
    play(sliding, reaches(80), slide_before)
    assert sliding.damage == 0


def test_jumping_does_not_clear_a_high_obstacle():
    game = g.Game(Empty([c.Obstacle(c.Kind.HIGH, 1, 40.0)]))
    play(game, reaches(80), jump_before(game, 40.0))
    assert game.damage == 1


def test_an_obstacle_across_two_paths_stops_a_runner_in_either():
    for lane in (1, 2):
        game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 1, 30.0, span=2)]))
        game.runner.lane = lane
        play(game, reaches(60))
        assert game.damage == 1, lane


def test_a_two_path_obstacle_leaves_the_other_path_clear():
    game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 1, 30.0, span=2)]))
    game.runner.lane = 0
    play(game, reaches(60))
    assert game.damage == 0


def test_an_obstacle_in_the_next_path_is_missed_by_a_runner_on_this_one():
    game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 2, 30.0), c.Obstacle(c.Kind.WALL, 0, 30.0)]))
    play(game, reaches(60))
    assert game.damage == 0


def test_a_coin_in_the_runners_path_is_taken_once_for_its_value():
    coin = c.Coin((0, 0), 1, 20.0, 0.5, 1)
    game = g.Game(Empty(coins=[coin]))
    heard = play(game, reaches(40))
    assert (game.coins, game.points) == (1, 1)
    assert heard.count(Event.COIN) == 1


def test_a_jewel_needs_a_jump_and_is_worth_its_value():
    jewel = c.Coin((0, 0), 1, 20.0, c.JEWEL_HEIGHT, 5)
    standing = g.Game(Empty(coins=[jewel]))
    play(standing, reaches(40))
    assert standing.coins == 0

    jumper = g.Game(Empty(coins=[jewel]))
    heard = play(jumper, reaches(40), lambda game, frame: game.runner.distance > 15.0 and game.runner.jump())
    assert (jumper.coins, jumper.points) == (1, 5)
    assert Event.JEWEL in heard


def test_a_coin_above_the_reach_of_a_standing_runner_is_not_taken():
    game = g.Game(Empty(coins=[c.Coin((0, 0), 1, 20.0, 2.2, 1)]))
    play(game, reaches(40))
    assert game.coins == 0


def test_a_coin_high_up_is_taken_in_a_jump():
    game = g.Game(Empty(coins=[c.Coin((0, 0), 1, 20.0, 2.2, 1)]))
    play(game, reaches(40), lambda game, frame: game.runner.distance > 14.0 and game.runner.jump())
    assert game.coins == 1


def test_a_coin_in_another_path_is_left_behind():
    game = g.Game(Empty(coins=[c.Coin((0, 0), 2, 20.0, 1.0, 1)]))
    play(game, reaches(40))
    assert game.coins == 0


def test_the_crashed_run_stands_still():
    game = g.Game(Empty(walls(30.0, 70.0, 110.0)))
    play(game, reaches(200))
    where = game.runner.distance
    game.advance(1.0)
    assert game.runner.distance == where


def test_meters_run_is_the_distance_in_whole_metres():
    game = g.Game(Empty())
    play(game, reaches(12.5))
    assert game.meters == int(game.runner.distance)


def test_a_long_step_is_taken_in_small_ones_so_nothing_is_stepped_over():
    game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 1, 30.0)]))
    game.runner.distance = 25.0
    heard = game.advance(1.0)
    assert Event.HIT in heard


def test_a_new_run_starts_clean():
    game = g.Game(Empty(walls(30.0, 70.0, 110.0)))
    play(game, reaches(200))
    game.restart()
    assert (game.state, game.runner.distance, game.coins, game.points, game.damage,
            game.crashed_into) == (g.State.RUNNING, 0.0, 0, 0, 0, None)


# -- damage ---------------------------------------------------------------------------

def walls(*distances, lane=1):
    return [c.Obstacle(c.Kind.WALL, lane, d) for d in distances]


def run_to(game, distance):
    """Run until the runner is ``distance`` down the hill; returns the events heard."""
    return play(game, lambda game: game.runner.distance >= distance
                or game.state is g.State.CRASHED)


def test_a_hit_takes_the_runners_speed_back_to_where_it_started_once_it_is_up_again():
    game = g.Game(Empty(walls(30.0)))
    game.runner.speed = 20.0
    play(game, lambda game: game.damage == 1)
    assert game.runner.speed == 0.0
    play(game, lambda game: game.state is g.State.RUNNING)
    assert game.runner.speed == pytest.approx(START_SPEED, abs=0.01)


def test_the_runner_speeds_up_again_after_a_hit():
    game = g.Game(Empty(walls(30.0)))
    game.runner.speed = 20.0
    play(game, lambda game: game.damage == 1)
    run_to(game, 200.0)
    assert START_SPEED < game.runner.speed < 20.0


# -- the fall -------------------------------------------------------------------------

def hit(game=None):
    game = game or g.Game(Empty(walls(30.0)))
    heard = play(game, lambda game: game.damage == 1)
    return game, heard


def test_a_hit_knocks_the_runner_down_and_it_stays_where_it_fell_while_it_gets_up():
    game, _ = hit()
    assert game.state is g.State.DOWN
    fell_at = game.runner.distance
    for _ in range(int(g.STUMBLE_TIME * 60) - 5):
        game.advance(FRAME)
    assert game.state is g.State.DOWN and game.runner.distance == fell_at


def test_the_runner_is_back_on_its_feet_after_the_stumble_and_runs_on():
    game, _ = hit()
    fell_at = game.runner.distance
    heard = play(game, lambda game: game.state is g.State.RUNNING)
    assert Event.RISE in heard
    run_to(game, fell_at + 5.0)
    assert game.state is g.State.RUNNING and game.runner.distance > fell_at + 4.0


def test_the_stumble_takes_as_long_as_it_says():
    game, _ = hit()
    frames = 0
    while game.state is g.State.DOWN:
        game.advance(FRAME)
        frames += 1
    assert frames * FRAME == pytest.approx(g.STUMBLE_TIME, abs=2 * FRAME)


def test_the_runner_is_knocked_back_to_before_the_obstacle_it_hit():
    obstacle = c.Obstacle(c.Kind.WALL, 1, 30.0)
    game, _ = hit(g.Game(Empty([obstacle])))
    assert game.runner.distance <= obstacle.distance - obstacle.depth / 2 - g.BODY_HALF_DEPTH


def test_nothing_the_runner_asks_for_is_done_while_it_is_down():
    game, _ = hit()
    for command in (lambda: game.runner.jump(), lambda: game.runner.slide(),
                    lambda: game.runner.hop(+1), lambda: game.runner.hop(-1)):
        assert command() is False
    assert game.runner.grounded and not game.runner.sliding and game.runner.drain_events() == []


def test_the_runner_can_act_again_once_it_is_up():
    game, _ = hit()
    play(game, lambda game: game.state is g.State.RUNNING)
    assert game.runner.jump() is True


def test_a_coin_is_not_taken_by_a_runner_who_is_down():
    coin = c.Coin((0, 0), 1, 30.0, 0.5, 1)
    game = g.Game(Empty(walls(30.0), [coin]))
    play(game, lambda game: game.damage == 1)
    for _ in range(int(g.STUMBLE_TIME * 60) - 10):
        game.advance(FRAME)
    assert game.state is g.State.DOWN and game.coins == 0


def test_the_hit_that_ends_the_run_is_a_fall_with_no_getting_up():
    game = g.Game(Empty(walls(30.0, 70.0, 110.0)))
    run_to(game, 300.0)
    assert game.state is g.State.CRASHED
    game.advance(g.STUMBLE_TIME * 2)
    assert game.state is g.State.CRASHED


def test_one_obstacle_is_one_hit_however_long_the_runner_is_in_it():
    game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 1, 30.0), c.Obstacle(c.Kind.LOW, 1, 33.0)]))
    heard = run_to(game, 90.0)
    assert (game.damage, heard.count(Event.HIT)) == (1, 1)


def test_the_third_hit_ends_the_run():
    game = g.Game(Empty(walls(30.0, 70.0, 110.0)))
    heard = run_to(game, 300.0)
    assert (game.damage, game.state) == (3, g.State.CRASHED)
    assert heard.count(Event.HIT) == 2 and heard.count(Event.CRASH) == 1


def test_two_hits_leave_the_runner_running():
    game = g.Game(Empty(walls(30.0, 70.0)))
    run_to(game, 300.0)
    assert (game.damage, game.state) == (2, g.State.RUNNING)


def test_the_runner_cannot_be_hit_again_while_it_is_recovering():
    game = g.Game(Empty(walls(30.0, 30.0 + g.RECOVERY_DISTANCE - 2.0)))
    run_to(game, 150.0)
    assert game.damage == 1


def test_an_obstacle_after_the_recovery_is_a_hit():
    game = g.Game(Empty(walls(30.0, 30.0 + g.RECOVERY_DISTANCE + 8.0)))
    run_to(game, 150.0)
    assert game.damage == 2


def test_five_hundred_metres_without_a_hit_heals_one_damage():
    game = g.Game(Empty(walls(30.0, 70.0)))
    run_to(game, 70.0 + g.HEAL_DISTANCE - 20.0)
    assert game.damage == 2
    heard = run_to(game, 70.0 + g.HEAL_DISTANCE + 20.0)
    assert (game.damage, heard.count(Event.HEAL)) == (1, 1)


def test_the_five_hundred_metres_are_counted_from_the_last_hit():
    game = g.Game(Empty(walls(30.0, 400.0)))
    run_to(game, 30.0 + g.HEAL_DISTANCE + 50.0)
    assert game.damage == 2
    run_to(game, 400.0 + g.HEAL_DISTANCE + 50.0)
    assert game.damage == 1


def test_every_five_hundred_metres_heals_another():
    game = g.Game(Empty(walls(30.0, 70.0)))
    run_to(game, 70.0 + 2 * g.HEAL_DISTANCE + 20.0)
    assert game.damage == 0


def test_a_runner_with_no_damage_has_nothing_to_heal_and_a_long_run_is_not_credit():
    game = g.Game(Empty(walls(1500.0)))
    heard = run_to(game, 1400.0)
    assert (game.damage, Event.HEAL in heard) == (0, False)
    run_to(game, 1600.0)
    assert game.damage == 1


# -- the body rising from a slide -----------------------------------------------------------

def slide_at(gap):
    """A game with a high barrier ``gap`` seconds ahead of a runner who slides now."""
    game = g.Game(Empty([c.Obstacle(c.Kind.HIGH, 1, 100.0)]))
    game.runner.distance = 100.0 - game.runner.speed * gap
    game.runner.slide()
    return game


def test_a_barrier_met_while_the_body_is_low_is_slid_under():
    from openglcontext_hill_runner.runner import SLIDE_RISE, SLIDE_TIME
    game = slide_at(SLIDE_TIME - SLIDE_RISE - 0.1)
    run_to(game, 130.0)
    assert game.damage == 0


def test_a_barrier_met_while_the_body_is_rising_is_a_hit():
    from openglcontext_hill_runner.runner import SLIDE_TIME
    game = slide_at(SLIDE_TIME - 0.05)
    run_to(game, 130.0)
    assert game.damage == 1 and game.crashed_into.kind is c.Kind.HIGH


def test_a_slide_held_on_goes_under_barrier_after_barrier():
    game = g.Game(Empty([c.Obstacle(c.Kind.HIGH, 1, d) for d in (60.0, 100.0, 140.0)]))
    game.runner.distance = 30.0
    play(game, lambda game: game.runner.distance > 170.0,
         lambda game, frame: game.runner.slide() if frame % 20 == 0 else None)
    assert game.damage == 0
