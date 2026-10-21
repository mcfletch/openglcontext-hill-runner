"""The hill's ground cover: grass to the edge of the trails, and the plants that grow with it.

The forest demo's art holds grasses, ferns, nettles, flowers and shrubs. They are drawn as two
nodes because the countries want them in different proportions: a meadow is grass with few
ferns, a forest the other way about. The grass is grown taller than the forest demo draws it,
so that near the trail it reads as grass seen from a runner's height and not as a lawn.
"""
import math
from typing import Any

import numpy as np
from OpenGLContext.scenegraph.vegetation.cover import (
    CARD_SUN,
    CoverSpecies,
    GroundCover,
    read_cover_manifest,
)
from OpenGLContext.scenegraph.vegetation.coverprocess import CoverProcess, CoverScatterMaker

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner.canopy import Canopy
from openglcontext_hill_runner.landscape import Landscape
from openglcontext_hill_runner.scenery import HillField

#: How many times taller the grass grows than the forest demo's art draws it.
TALL = 2.2
#: How far the cover reaches, in metres: the geometry, the cards that replace it, and no coarse
#: field beyond those. The haze hides everything past ``app.HAZE_RANGE``, and the sparse far field
#: reads on the horizon as dark rocks that vanish as they are neared.
CLUMP_REACH = 34.0
CARD_REACH = 130.0
#: Where each rung of the cover dissolves into the next begins, as a fraction of its outer edge. At
#: 24 m/s the default window of a fifth of the radius is crossed in a third of a second and
#: reads as a pop.
FADE = 0.5
#: How far down the trail ahead the cover is scattered before the runner gets there, in metres,
#: the spacing of the places it is scattered about, and how far the runner goes between times.
PREPARE_AHEAD = 210.0
PREPARE_STEP = 30.0
PREPARE_EVERY = 20.0
#: Metres from the middle of a trail inside which the grass does not grow, and the width of the
#: band over which it thickens to full; the same for the other plants. A trail is 0.5 m wide.
GRASS_CLEAR = (0.32, 0.25)
PLANTS_CLEAR = (0.55, 0.4)


def split(species: list[CoverSpecies]) -> tuple[list[CoverSpecies], list[CoverSpecies]]:
    """``species`` as the grasses, grown :data:`TALL`, and the plants that are not grass.

    Every card is lit as much as the lit geometry it stands in for (``CARD_SUN``), whatever light
    the pack records, so the ground does not change brightness where the cards give way to
    the geometry.
    """
    lit = [s.varied(sunLevel=CARD_SUN) for s in species]
    grasses = [s.varied(height=s.height * TALL) for s in lit if s.name.startswith('grass')]
    return grasses, [s for s in lit if not s.name.startswith('grass')]


def _weight(land: Landscape, density: Any, clear: tuple[float, float]) -> Any:
    """How much grows at ``(x, z)``: nothing near a trail, thickening to the country's density."""
    start, band = clear

    def weight(x: Any, z: Any) -> Any:
        x, d = np.asarray(x, dtype=float), -np.asarray(z, dtype=float)
        free = np.clip((land.trail_gap(x, d) - start) / band, 0.0, 1.0)
        return free * np.minimum(1.0, density(x, d))
    return weight


def grass_where(land: Landscape) -> Any:
    """How much grass grows at ``(x, z)``, from 0 to 1."""
    return _weight(land, land.grass_density, GRASS_CLEAR)


def plants_where(land: Landscape) -> Any:
    """How much of the ferns, nettles, flowers and shrubs grows at ``(x, z)``, from 0 to 1."""
    return _weight(land, land.cover_density, PLANTS_CLEAR)


#: The factory a scatter process builds its :class:`CoverScatterMaker` with, as the engine names it.
SCATTER = 'openglcontext_hill_runner.groundcover:scatter_maker'


def grown(art: str, land: Landscape, reach: float, canopy: Canopy,
          scatter_only: bool = False) -> tuple[GroundCover, GroundCover]:
    """The grass and the other plants of ``land``, as two ground covers.

    ``scatter_only`` leaves out the clump meshes, which only a node that draws needs.
    """
    grasses, plants = split(read_cover_manifest(art))
    if scatter_only:
        grasses, plants = ([s.varied(clump='') for s in grasses],
                           [s.varied(clump='') for s in plants])
    ground = HillField(land)
    grass = GroundCover(ground, grasses, clump_radius=reach, card_radius=CARD_REACH,
                        far_radius=CARD_REACH, fade=FADE, mask=grass_where(land),
                        shade=canopy, background=not scatter_only)
    other = GroundCover(ground, plants, clump_radius=reach, card_radius=CARD_REACH,
                        far_radius=CARD_REACH, fade=FADE, mask=plants_where(land),
                        shade=canopy, background=not scatter_only)
    return grass, other


def scatter_maker(seed: int, caves: bool, art: str, reach: float, which: str) -> CoverScatterMaker:
    """What a scatter process holds: the ground cover of the hill of ``seed`` that grows
    ``which`` ('grass' or 'plants'), with a ``make`` for its scatter."""
    land = Landscape(c.Course(seed, caves))
    grass, plants = grown(art, land, reach, Canopy(land), scatter_only=True)
    return CoverScatterMaker({'grass': grass, 'plants': plants}[which])


class Cover:
    """The grass and the other plants of one hill, scattered about the camera on a worker.

    ``reach`` is how far from the camera, in metres, plants are drawn as geometry; beyond it, as
    far as :data:`CARD_REACH`, they are cards. Geometry is what costs the GPU most, and a shorter
    reach is the cheapest way to buy back a frame.

    With ``processes`` the scatter is made by a process for each node rather than by threads of
    the game's own, which would take the interpreter lock from the frame; the nodes scatter in
    line until the processes are up.
    """

    def __init__(self, art: str, land: Landscape, reach: float = CLUMP_REACH,
                 processes: bool = False) -> None:
        self.art = art
        self.reach = reach
        self.processes = processes
        self.land = land
        #: How much of the sun reaches the ground between the trees: what each plant is lit by.
        self.canopy = Canopy(land)
        self._prepared_at = -math.inf
        self.grass, self.plants = grown(art, land, reach, self.canopy)
        self._scatter_in_processes()

    def _scatter_in_processes(self) -> None:
        """Hand each node's scatter to a process of its own, if there are to be any."""
        if not self.processes:
            return
        course = self.land.course
        for which, node in (('grass', self.grass), ('plants', self.plants)):
            node.use_process(CoverProcess(
                SCATTER, (course.seed, course.caves, self.art, self.reach, which)))

    @property
    def nodes(self) -> list[GroundCover]:
        """The nodes to put in the scene."""
        return [self.grass, self.plants]

    def relocate(self, land: Landscape) -> None:
        """Grow the same cover on another hill, keeping what the nodes hold on the GPU."""
        self.land = land
        self.canopy = Canopy(land)
        self._prepared_at = -math.inf
        ground = HillField(land)
        self.grass.relocate(ground, mask=grass_where(land), shade=self.canopy)
        self.plants.relocate(ground, mask=plants_where(land), shade=self.canopy)
        self._scatter_in_processes()

    def fill(self, eye: Any) -> None:
        """Scatter the cover about ``eye`` now, so the first frame has it."""
        x, z = float(eye[0]), float(eye[2])
        for node in self.nodes:
            node.apply_near(node.compute_near(x, z))
            node.apply_far(node.compute_far(x, z))
            node.update(eye)

    def update(self, eye: Any) -> None:
        """Keep the cover about the camera at ``eye``, and scatter the trail ahead of it as it
        runs; call once a frame."""
        for node in self.nodes:
            node.update(eye)
        run = -float(eye[2])
        if not 0.0 <= run - self._prepared_at < PREPARE_EVERY:
            self._prepared_at = run
            ahead = [run + step for step in np.arange(PREPARE_STEP, PREPARE_AHEAD + 1.0, PREPARE_STEP)]
            places = [(float(self.land.course.centre(d)), -d) for d in ahead]
            for node in self.nodes:
                node.prepare(places)

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the workers to finish a scatter; whether both did in time."""
        return all([node.wait(timeout) for node in self.nodes])

    def shutdown(self) -> None:
        """Stop the workers, and the processes if there are any."""
        for node in self.nodes:
            node.shutdown()
