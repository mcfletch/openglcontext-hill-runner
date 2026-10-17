"""The rules: collisions, treasure, and the end of a run."""
import pytest

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner.events import Event

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
    for frame in range(60 * 30):
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


def test_running_into_a_wall_ends_the_run():
    game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 1, 30.0)]))
    heard = play(game, reaches(60))
    assert game.state is g.State.CRASHED
    assert Event.CRASH in heard
    assert game.crashed_into.kind is c.Kind.WALL


def test_a_hop_clears_the_wall():
    game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 1, 30.0)]))
    play(game, reaches(60), lambda game, frame: frame == 5 and game.runner.hop(+1))
    assert game.state is g.State.RUNNING


def test_a_low_obstacle_is_jumped_over_and_stops_a_runner_who_stays_down():
    low = c.Obstacle(c.Kind.LOW, 1, 40.0)
    ran_into = g.Game(Empty([low]))
    play(ran_into, reaches(80))
    assert ran_into.state is g.State.CRASHED

    jumper = g.Game(Empty([low]))
    play(jumper, reaches(80), jump_before(jumper, 40.0))
    assert jumper.state is g.State.RUNNING


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
    assert standing.state is g.State.CRASHED

    sliding = g.Game(Empty([high]))

    def slide_before(game, frame):
        if 40.0 - game.runner.distance <= game.runner.speed * 0.2:
            game.runner.slide()
    play(sliding, reaches(80), slide_before)
    assert sliding.state is g.State.RUNNING


def test_jumping_does_not_clear_a_high_obstacle():
    game = g.Game(Empty([c.Obstacle(c.Kind.HIGH, 1, 40.0)]))
    play(game, reaches(80), jump_before(game, 40.0))
    assert game.state is g.State.CRASHED


def test_an_obstacle_across_two_paths_stops_a_runner_in_either():
    for lane in (1, 2):
        game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 1, 30.0, span=2)]))
        game.runner.lane = lane
        play(game, reaches(60))
        assert game.state is g.State.CRASHED, lane


def test_a_two_path_obstacle_leaves_the_other_path_clear():
    game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 1, 30.0, span=2)]))
    game.runner.lane = 0
    play(game, reaches(60))
    assert game.state is g.State.RUNNING


def test_an_obstacle_in_the_next_path_is_missed_by_a_runner_on_this_one():
    game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 2, 30.0), c.Obstacle(c.Kind.WALL, 0, 30.0)]))
    play(game, reaches(60))
    assert game.state is g.State.RUNNING


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
    game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 1, 30.0)]))
    play(game, reaches(60))
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
    assert Event.CRASH in heard


def test_a_new_run_starts_clean():
    game = g.Game(Empty([c.Obstacle(c.Kind.WALL, 1, 30.0)]))
    play(game, reaches(60))
    game.restart()
    assert (game.state, game.runner.distance, game.coins, game.points,
            game.crashed_into) == (g.State.RUNNING, 0.0, 0, 0, None)
