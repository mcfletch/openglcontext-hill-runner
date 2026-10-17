"""An autopilot that runs the course through the commands a player has.

It looks a short way down the hill, and for the obstacle ahead in its own path
hops, jumps or slides at the moment that clears it. It drives the attract mode,
and the test that every generated course can be run.
"""
from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner.runner import HOP_TIME

#: Metres looked ahead.
LOOK = 40.0
#: Seconds before the obstacle's near edge that each command is given: a jump
#: peaks 0.3 s in, so it is over the obstacle; a slide of 0.8 s covers its depth.
JUMP_LEAD = 0.27
#: A crevice is crossed from nearer to it: the jump is then over its far side when it comes down.
CREVICE_LEAD = 0.20
SLIDE_LEAD = 0.20
HOP_LEAD = 0.80
#: Metres of clearance a hop keeps from an obstacle in a path it crosses, and seconds of room
#: it leaves to deal with one in the path it ends in.
HOP_MARGIN = 0.8
ARRIVAL_ROOM = 0.4
#: Obstacles this close together in distance stand in the same row.
SAME_ROW = 3.0
#: How far into a fork a pilot still hops toward its new paths: they part beyond reach soon after.
FORK_WINDOW = 24.0


class Pilot:
    """Drives one :class:`~openglcontext_hill_runner.game.Game`."""

    def __init__(self, game: g.Game, branch: "bool | None" = None) -> None:
        """``branch`` says whether to take the forks: always, never, or by turns (None)."""
        self.game = game
        self.branch = branch

    def drive(self) -> None:
        """Give the runner the commands this moment calls for."""
        runner = self.game.runner
        course = self.game.course
        ahead = course.obstacles_between(runner.distance - 1.0, runner.distance + LOOK)
        mine = [o for o in ahead if runner.lane in o.lanes]
        if not mine:
            self._take_the_fork()
            return
        nearest = mine[0]
        gap = nearest.distance - nearest.depth / 2 - g.BODY_HALF_DEPTH - runner.distance
        if gap < -nearest.depth:
            return
        if nearest.kind is c.Kind.WALL:
            if gap <= runner.speed * HOP_LEAD:
                self._hop_clear_of(nearest, ahead)
        elif nearest.kind in (c.Kind.LOW, c.Kind.CREVICE):
            lead = CREVICE_LEAD if nearest.kind is c.Kind.CREVICE else JUMP_LEAD
            # The course puts a jump only where no low ceiling is near; the pilot checks anyway.
            if (gap <= runner.speed * lead and runner.grounded
                    and self.game.course.jump_clear(nearest.distance)):
                runner.jump()
        elif gap <= runner.speed * SLIDE_LEAD:
            runner.slide()

    def _takes(self, fork: c.Fork) -> bool:
        return fork.index % 2 == 0 if self.branch is None else self.branch

    def _take_the_fork(self) -> None:
        """While a fork is just starting, hop toward its new paths, or off them if declining."""
        runner = self.game.runner
        course = self.game.course
        fork = course.fork_at(runner.distance)
        if fork is None or runner.distance > fork.start + FORK_WINDOW:
            return
        want = fork.lanes[0] if self._takes(fork) else c.LANE_COUNT // 2
        if want in course.lanes_at(runner.distance) and runner.lane != want:
            runner.hop(1 if want > runner.lane else -1)

    def _reachable(self) -> list[int]:
        """The paths a run of hops gets to from the one the runner is on, before the row."""
        runner = self.game.runner
        course = self.game.course
        window = runner.speed * HOP_TIME * 2
        found, edge = [runner.lane], [runner.lane]
        while edge:
            lane = edge.pop()
            for neighbour in (lane - 1, lane + 1):
                if (neighbour not in found and neighbour in course.lanes_at(runner.distance)
                        and not course.blocks_hop(lane, neighbour, runner.distance,
                                                  runner.distance + window)):
                    found.append(neighbour)
                    edge.append(neighbour)
        return found

    def _hop_clear_of(self, wall: c.Obstacle, ahead: list[c.Obstacle]) -> None:
        """Hop toward the nearest path the row's walls leave open, and the emptiest if two tie.

        That may be two paths away: the hops are made one after the other, well before the
        row, so the runner is never across a wall as it passes.
        """
        runner = self.game.runner
        walled = {lane for o in ahead if o.kind is c.Kind.WALL
                  and abs(o.distance - wall.distance) < SAME_ROW for lane in o.lanes}
        reachable = self._reachable()
        options = [(abs(lane - runner.lane), sum(1 for o in ahead if lane in o.lanes), lane)
                   for lane in reachable if lane not in walled]
        if not options:
            return
        target = min(options)[2]
        direction = 1 if target > runner.lane else -1
        path = range(runner.lane + direction, target + direction, direction)
        travel = runner.speed * HOP_TIME * len(path)
        for lane in path:
            for o in ahead:
                if lane not in o.lanes or (o.kind is c.Kind.WALL
                                           and abs(o.distance - wall.distance) < SAME_ROW):
                    continue
                # An obstacle in a lane crossed, or one the runner would arrive too close to
                # to deal with, means waiting a moment for it to pass.
                if (o.distance + o.depth / 2 + HOP_MARGIN >= runner.distance
                        and o.distance - o.depth / 2 - HOP_MARGIN
                        <= runner.distance + travel + runner.speed * ARRIVAL_ROOM):
                    return
        runner.hop(direction)
