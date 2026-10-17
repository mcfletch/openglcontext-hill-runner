"""The treasure on the hill: which coins are in view, and the nodes that show them.

A coin disappears when it is taken, so coins are not part of the static tiles.
They live in one :class:`TreasureField`, which keeps a node for each coin in view
and lets the rest go.
"""
import math

import numpy as np
from OpenGLContext.scenegraph import surfaces as sf
from OpenGLContext.scenegraph.basenodes import Group, Shape, Transform

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner.look import Look, make_look

#: Metres of the hill shown ahead of the runner, and behind it.
AHEAD = 220.0
BEHIND = 12.0
#: How far a coin rises and falls, in metres, and how fast it turns, in radians a second.
BOB = 0.05
SPIN = 2.6
COIN_RADIUS = 0.125
JEWEL_RADIUS = 0.13
JEWEL_CROWN = 0.07
JEWEL_PAVILION = 0.15


def coins_in_view(game: g.Game) -> list[c.Coin]:
    """The coins between a little behind the runner and the view ahead, not yet taken."""
    where = game.runner.distance
    return [coin for coin in game.course.coins_between(where - BEHIND, where + AHEAD)
            if not game.taken(coin)]


def coin_geometry() -> sf.Geometry:
    """A coin standing on its edge, its faces to the front and the back.

    The faces carry the coin's picture edge to edge; the rim is textured from the plain
    gold outside the picture's circle.
    """
    thick = 0.025
    rim = sf.placed(sf.cylinder(COIN_RADIUS, thick, sides=32, repeat=1.0),
                    rotation=(1.0, 0.0, 0.0, math.pi / 2))
    rim = rim._replace(texcoords=np.tile(np.array([0.5, 0.01], 'f'), (len(rim.positions), 1)))
    faces = [sf.placed(sf.polygon(COIN_RADIUS, sides=32, repeat=2.0 * COIN_RADIUS),
                       (0.0, 0.0, side * thick / 2),
                       (0.0, 1.0, 0.0, 0.0 if side > 0 else math.pi)) for side in (-1, 1)]
    return sf.merge([rim, *faces])


def jewel_geometry() -> sf.Geometry:
    """A jewel cut with a table, a crown and a pavilion, in flat facets."""
    sides = 8
    angle = 2.0 * math.pi * np.arange(sides) / sides
    ring = np.stack([np.cos(angle), np.zeros(sides), np.sin(angle)], axis=-1)
    girdle = ring * JEWEL_RADIUS
    table = ring * JEWEL_RADIUS * 0.58 + (0.0, JEWEL_CROWN, 0.0)
    top = np.array([0.0, JEWEL_CROWN, 0.0])
    point = np.array([0.0, -JEWEL_PAVILION, 0.0])
    triangles = []
    for i in range(sides):
        j = (i + 1) % sides
        triangles += [(girdle[i], girdle[j], table[j]), (girdle[i], table[j], table[i]),
                      (table[i], table[j], top), (girdle[j], girdle[i], point)]
    positions, normals = [], []
    middle = np.array([0.0, (JEWEL_CROWN - JEWEL_PAVILION) / 2.0, 0.0])
    for p, q, r in triangles:
        normal = np.cross(q - p, r - p)
        normal /= np.linalg.norm(normal)
        if normal @ ((p + q + r) / 3.0 - middle) < 0.0:
            q, r, normal = r, q, -normal
        positions += [p, q, r]
        normals += [normal] * 3
    positions_array = np.array(positions, 'f')
    count = len(positions_array)
    return sf.Geometry(positions_array, np.array(normals, 'f'),
                       (positions_array[:, [0, 2]] / (2.0 * JEWEL_RADIUS) + 0.5).astype('f'),
                       np.tile(np.array([1.0, 0.0, 0.0, 1.0], 'f'), (count, 1)),
                       np.arange(count, dtype=np.uint32))


class TreasureField(Group):
    """A node for each coin or jewel in view, turning as it hangs above the ground."""

    def __init__(self, look: Look | None = None) -> None:
        super().__init__()
        look = look or make_look()
        self._coin = self._shaped(coin_geometry(), look.coin)
        self._jewel = self._shaped(jewel_geometry(), look.jewel)
        self._nodes: dict[tuple[int, int], Transform] = {}
        self._coins: dict[tuple[int, int], c.Coin] = {}

    @staticmethod
    def _shaped(geometry: sf.Geometry, material: object) -> Shape:
        shape: Shape = sf.shape(geometry, material).children[0]
        return shape

    def shape_for(self, coin: c.Coin) -> Shape:
        """The shape every coin of this kind shares."""
        return self._jewel if coin.value > 1 else self._coin

    def update(self, game: g.Game, now: float) -> None:
        """Show the coins in view as of ``now`` seconds, and let go of the rest."""
        wanted = {coin.ident: coin for coin in coins_in_view(game)}
        # A coin of another course can have the name a coin of this one has, so what is held
        # is compared by what the coin is and not by its name.
        stale = [ident for ident, coin in self._coins.items() if wanted.get(ident) != coin]
        changed = bool(stale) or wanted.keys() != self._nodes.keys()
        for ident in stale:
            del self._nodes[ident]
            del self._coins[ident]
        for ident, coin in wanted.items():
            if ident not in self._nodes:
                self._nodes[ident] = Transform(children=[self.shape_for(coin)])
                self._coins[ident] = coin
        for ident, node in self._nodes.items():
            coin = self._coins[ident]
            phase = coin.distance * 0.7
            node.translation = (game.course.lane_x(coin.lane, coin.distance),
                                float(c.ground_height(coin.distance)) + coin.height
                                + BOB * math.sin(now * 2.5 + phase),
                                -coin.distance)
            node.rotation = (0.0, 1.0, 0.0, (now * SPIN + phase) % (2.0 * math.pi))
        if changed:
            self.children = list(self._nodes.values())
