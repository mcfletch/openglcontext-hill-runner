"""The window: draws a :class:`~openglcontext_hill_runner.session.Session`.

The rules, the pilot, the cues and the camera's target are in the session and
the modules under it. This class builds the scene, feeds the session the keys
and the time that passed, and applies what a frame says to the model, the
mixer, the HUD and the camera.
"""
import contextlib
import importlib.resources as resources
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
from OpenGLContext.audio import scene as audioscene
from OpenGLContext.character import CharacterModel
from OpenGLContext.context import Context, getCurrentContext
from OpenGLContext.contextdefinition import ContextDefinition
from OpenGLContext.events.systemtime import systemTime
from OpenGLContext.move.followcam import FollowCamera
from OpenGLContext.scenegraph.basenodes import (
    Background,
    DirectionalLight,
    Fog,
    Transform,
    sceneGraph,
)
from OpenGLContext.scenegraph.generatedtiles import GeneratedTiles
from OpenGLContext.scenegraph.vegetation.cover import GroundCover, read_cover_manifest
from OpenGLContext.ui import dialogs, settings

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import credits, menu, woods
from openglcontext_hill_runner.hud import RunHud
from openglcontext_hill_runner.landscape import Landscape
from openglcontext_hill_runner.look import make_look
from openglcontext_hill_runner.runner import HOP_TIME, MAX_SPEED
from openglcontext_hill_runner.scenery import HillField
from openglcontext_hill_runner.session import Frame, Session
from openglcontext_hill_runner.sound import Sounds
from openglcontext_hill_runner.tiles import HillTiles
from openglcontext_hill_runner.treasure import TreasureField

#: How far ahead of the camera the hill is made, in metres.
REACH = 260.0
FOV = math.radians(62.0)
#: The camera's offset from the point it looks at, and how far it backs off at full speed.
CAMERA_OFFSET = (0.0, 1.85, 5.1)
CAMERA_PULL_BACK = 0.08
#: The sun's direction: from behind the runner and a little to the left, so shadows fall ahead
#: along the trails, and a coin's shadow lies on its own trail.
SUN = (-0.18, -0.75, -0.62)
#: How much of the hill's slope the runner leans into, and how far it leans into a hop.
SLOPE_LEAN = 0.6
HOP_LEAN = 0.35
#: Cover stays this far from the middle of the nearest path, which keeps it off the gaps too.
TRAIL_CLEAR = 0.55
SKY = [(0.26, 0.48, 0.84), (0.55, 0.74, 0.94), (0.84, 0.90, 0.93)]
HAZE = (0.80, 0.87, 0.92)


def _in_the_sun(x: Any, z: Any) -> Any:  # noqa: ARG001 the engine asks for a value at (x, z)
    """How much sun reaches the cover: all of it, for each point."""
    return np.ones_like(np.asarray(x, dtype=float))


def _cover_where(land: Landscape) -> Any:
    """How much ground cover grows at ``(x, z)``: none on the trails and the gaps between them,
    and as much beside them as the country has."""
    def weight(x: Any, z: Any) -> Any:
        x, d = np.asarray(x, dtype=float), -np.asarray(z, dtype=float)
        clear = np.clip((land.trail_gap(x, d) - TRAIL_CLEAR) / 0.4, 0.0, 1.0)
        return clear * np.minimum(1.0, land.cover_density(x, d))
    return weight


class RunnerContext(Context):
    """The runner game's window."""

    windowSystemName = 'glfw'
    #: No free-fly: the chase camera is all that moves the view.
    contextDefinition = ContextDefinition(navigation=None)

    #: Set by ``run.main`` from the command line.
    seed = 1
    demo = False
    sound = True
    photos = True
    forest = True

    # -- building ----------------------------------------------------------
    def OnInit(self) -> None:
        self.look = make_look(photos=self.photos)
        path = resources.files('openglcontext_hill_runner') / 'assets' / 'runner.glb'
        self.model = CharacterModel.load(str(path))
        self.mixer = Sounds(audioscene.engine_for(self) if self.sound else None)
        self.hud = RunHud()
        self.addHUDLayer(self.hud.layer)
        self.treasure = TreasureField(self.look)
        self.woods: woods.Woods | None = None
        self.cover: GroundCover | None = None
        self.camera = FollowCamera(self.platform, offset=CAMERA_OFFSET,
                                   pull_back=CAMERA_PULL_BACK, pull_back_speed=MAX_SPEED)
        self.platform.setFrustum(fieldOfView=FOV, near=0.3, far=REACH + 150.0)
        self._build_runner_node()
        self.terrain: GeneratedTiles | None = None
        self.sg = sceneGraph(children=[])
        self._start(self.seed)
        for name, command in (('<left>', self._hop_left), ('<right>', self._hop_right),
                              ('<up>', self._jump), ('<down>', self._slide),
                              ('<escape>', self._pause)):
            self.addEventHandler('keyboard', name=name, state=1, function=command)
        self.mixer.start_music()
        print('\n'.join(credits.startup(self.woods is not None, self.photos)))

    def _plant_the_forest(self, land: Landscape) -> None:
        """Trees and ground cover from the forest demo's art, where it is installed, for ``land``."""
        art = woods.forest_art() if self.forest else None
        if self.cover is not None:
            self.cover.shutdown()
        self.cover = None
        if art is None:
            self.woods = None
            return
        if self.woods is None:
            self.woods = woods.Woods(woods.species(art), land)
        else:
            self.woods.relocate(land)
        self.cover = GroundCover(HillField(land), read_cover_manifest(art),
                                 mask=_cover_where(land), shade=_in_the_sun, background=True)

    def _build_runner_node(self) -> None:
        """The runner: placed, tipped to the slope, leaned into hops, turned to face down the hill."""
        facing = Transform(rotation=(0.0, 1.0, 0.0, math.pi), children=[self.model.group])
        self._lean = Transform(children=[facing])
        self._pitch = Transform(children=[self._lean])
        self._yaw = Transform(children=[self._pitch])
        self.runner_node = Transform(children=[self._yaw])

    def _start(self, seed: int) -> None:
        """Begin a run of the course ``seed`` makes."""
        with self._gl_current():
            if self.terrain is not None:
                self.terrain.dispose()
                self.terrain.shutdown()
            course = c.Course(seed)
            self.land = Landscape(course)
            self._plant_the_forest(self.land)
        self.session = Session(course, demo=self.demo)
        self.terrain = GeneratedTiles(
            HillTiles(self.land, self.look, trees=self.woods is None), reach=REACH, fovy=FOV)
        self._clip = ''
        self._game_over_shown = False
        self._last_x = self.session.game.runner.x
        self._last = systemTime()
        frame = self.session.step(0.0)
        self._look_through(frame, 0.0)
        self.terrain.prime(self._eye(), self._viewport_height())
        self._fill_the_cover()
        self.sg.children = [
            Background(skyColor=SKY, skyAngle=[1.0, 1.5], groundColor=[HAZE]),
            Fog(color=HAZE, visibilityRange=REACH * 1.15),
            DirectionalLight(direction=SUN, color=(1.0, 0.96, 0.88), intensity=2.4),
            self.terrain, self.treasure, self.runner_node,
            *[node for node in (self.woods, self.cover) if node is not None],
        ]
        self.mark('run-started', seed=seed, demo=bool(self.demo))

    @contextlib.contextmanager
    def _gl_current(self) -> Iterator[None]:
        """Run with this context's GL current: as it already is inside a draw or ``OnInit``, and
        as it is not in ``OnIdle`` or an event."""
        if getCurrentContext() is not None:
            yield
            return
        self.setCurrent()
        try:
            yield
        finally:
            self.unsetCurrent()

    def _eye(self) -> Any:
        """Where the camera is, as x, y, z."""
        return self.platform.position[:3]

    def _fill_the_cover(self) -> None:
        """Scatter the ground cover around the start before the first frame.

        After this the scatter runs on the cover's own worker as the runner moves.
        """
        if self.cover is None:
            return
        eye = self._eye()
        self.cover.apply_near(self.cover.compute_near(float(eye[0]), float(eye[2])))
        self.cover.apply_far(self.cover.compute_far(float(eye[0]), float(eye[2])))
        self.cover.update(eye)

    def _viewport_height(self) -> float:
        return float(max(1, self.getViewPort()[1]))

    # -- commands: the whole control scheme --------------------------------
    def _hop_left(self, event: Any) -> None:  # noqa: ARG002 OpenGLContext key-event callback signature
        self.session.game.runner.hop(-1)

    def _hop_right(self, event: Any) -> None:  # noqa: ARG002 OpenGLContext key-event callback signature
        self.session.game.runner.hop(+1)

    def _jump(self, event: Any) -> None:  # noqa: ARG002 OpenGLContext key-event callback signature
        self.session.game.runner.jump()

    def _slide(self, event: Any) -> None:  # noqa: ARG002 OpenGLContext key-event callback signature
        self.session.game.runner.slide()

    def _pause(self, event: Any) -> None:  # noqa: ARG002 OpenGLContext key-event callback signature
        if not self.overlays.visible:
            self.pushOverlay(menu.pause_panel(
                on_resume=self._resume, on_restart=self._restart,
                on_settings=self._settings, on_credits=self._credits, on_quit=self.OnQuit))

    def _credits(self) -> None:
        self.pushOverlay(dialogs.notice(
            'Credits', credits.text(self.woods is not None, self.photos), columns=64))

    def _settings(self) -> None:
        settings.open_settings(self)

    def _resume(self) -> None:
        self._last = systemTime()

    def _restart(self) -> None:
        self.seed += 1
        self._start(self.seed)

    # -- the frame ------------------------------------------------------------
    def OnIdle(self, *arguments: Any) -> Any:  # noqa: ARG002 OpenGLContext OnIdle callback signature
        now = systemTime()
        dt = min(now - self._last, 0.05)
        self._last = now
        paused = self.overlays.visible
        self.mixer.pause(paused)
        if not paused:
            frame = self.session.step(dt)
            self._apply(frame, dt, now)
            if self.session.over and not self._game_over_shown:
                self._game_over_shown = True
                self.pushOverlay(menu.game_over_panel(
                    self.session.game, on_retry=self._restart, on_quit=self.OnQuit))
        self.triggerRedraw(1)
        return 1

    def _apply(self, frame: Frame, dt: float, now: float) -> None:
        game = self.session.game
        self.mixer.play(frame.sounds)
        pose = frame.pose
        self.model.play(pose.clip, loop=pose.loop, fade=pose.fade, speed=pose.speed,
                        restart=pose.clip != self._clip)
        self._clip = pose.clip
        self.model.update(dt)
        self._place_runner(dt)
        self.treasure.update(game, now)
        self.hud.update(game)
        self._look_through(frame, dt)
        self._stream()

    def _stream(self) -> None:
        """Make, load and drop tiles with the GL context current: a dropped tile's buffers are deleted here."""
        if self.terrain is None:
            return
        with self._gl_current():
            eye = self._eye()
            self.terrain.update_for_camera(eye, self._viewport_height())
            if self.woods is not None:
                self.woods.update(eye)
            if self.cover is not None:
                self.cover.update(eye)

    def _place_runner(self, dt: float) -> None:
        runner = self.session.game.runner
        d = runner.distance
        self.runner_node.translation = (
            runner.x, float(self.land.height(runner.x, d)) + runner.height, -d)
        self._yaw.rotation = (0.0, 1.0, 0.0, -self.session.chase.heading())
        fall = (float(c.ground_height(d - 1.0)) - float(c.ground_height(d + 1.0))) / 2.0
        self._pitch.rotation = (1.0, 0.0, 0.0, -math.atan(fall) * SLOPE_LEAN)
        sideways = (runner.x - self._last_x) / dt if dt > 0 else 0.0
        self._last_x = runner.x
        across = c.LANE_WIDTH / HOP_TIME
        self._lean.rotation = (0.0, 0.0, 1.0, -HOP_LEAN * max(-1.0, min(1.0, sideways / across)))

    def _look_through(self, frame: Frame, dt: float) -> None:
        turn = self.session.chase.heading()
        back = CAMERA_OFFSET[2]
        self.camera.offset = np.array([-back * math.sin(turn), CAMERA_OFFSET[1],
                                       back * math.cos(turn)])
        self.camera.target(frame.target)
        self.camera.advance(dt, self.session.game.runner.speed)
        self.camera.apply()

    def OnQuit(self, event: Any = None) -> None:
        self.mixer.stop()
        if self.cover is not None:
            self.cover.shutdown()
        if self.woods is not None:
            self.woods.shutdown()
        if self.terrain is not None:
            self.terrain.shutdown()
        super().OnQuit(event)
