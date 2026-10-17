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
from OpenGLContext.ui import dialogs, settings

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import credits, menu, woods
from openglcontext_hill_runner.chase import CAMERA_BACK, CAMERA_UP
from openglcontext_hill_runner.groundcover import Cover
from openglcontext_hill_runner.hud import RunHud
from openglcontext_hill_runner.landscape import Landscape
from openglcontext_hill_runner.look import make_look
from openglcontext_hill_runner.runner import MAX_SPEED
from openglcontext_hill_runner.session import Frame, Session
from openglcontext_hill_runner.sound import Sounds
from openglcontext_hill_runner.tiles import HillTiles
from openglcontext_hill_runner.treasure import TreasureField

#: How far ahead of the camera the hill is made, in metres.
REACH = 260.0
#: The distance at which the haze hides everything: no further than the tiles are made, so the
#: far edge of the hill, and the trees standing beyond it, are lost in the haze rather than seen.
HAZE_RANGE = REACH
FOV = math.radians(62.0)
#: How much further the camera backs off at full speed, as a fraction of its distance.
CAMERA_PULL_BACK = 0.08
#: The sun's direction: from behind the runner and a little to the left, so shadows fall ahead
#: along the trails, and a coin's shadow lies on its own trail.
SUN = (-0.18, -0.75, -0.62)
#: The game's keys, each with the method of the window that answers it.
CONTROLS = {'<left>': '_hop_left', '<right>': '_hop_right', '<up>': '_jump',
            '<down>': '_slide', '<escape>': '_pause'}
SKY = [(0.26, 0.48, 0.84), (0.55, 0.74, 0.94), (0.84, 0.90, 0.93)]
HAZE = (0.80, 0.87, 0.92)


class RunnerContext(Context):
    """The runner game's window."""

    windowSystemName = 'glfw'
    #: No free-fly: the chase camera is all that moves the view.
    #: The title names the window and the pictures F2 saves (``Hill-Runner-0001.png``).
    #: Shadows are asked for outright, with a fixed three cascades so the runner's and the
    #: logs' shadows keep their detail whatever the frame rate. Frames wait for the display's
    #: refresh, which leaves the threads that scatter the cover and make the tiles their share
    #: of the interpreter between frames.
    contextDefinition = ContextDefinition(navigation=None, title='Hill Runner', shadows=True,
                                          shadowCascades=3, vsync=True)

    #: Set by ``run.main`` from the command line.
    seed = 1
    demo = False
    sound = True
    photos = True
    forest = True
    #: Whether the game opens on the main screen rather than running at once.
    main_menu = False

    #: The game opens on its own screen; Alt+F brings up the engine's developer overlay.
    debugOverlayStartsVisible = False

    # -- building ----------------------------------------------------------
    def OnInit(self) -> None:
        """Load the character and the materials, build the scene and start the first run."""
        self.look = make_look(photos=self.photos)
        path = resources.files('openglcontext_hill_runner') / 'assets' / 'runner.glb'
        self.model = CharacterModel.load(str(path))
        self.mixer = Sounds(audioscene.engine_for(self) if self.sound else None)
        self.hud = RunHud()
        self.addHUDLayer(self.hud.layer)
        self.treasure = TreasureField(self.look)
        self.woods: woods.Woods | None = None
        self.cover: Cover | None = None
        self.camera = FollowCamera(self.platform, offset=(0.0, CAMERA_UP, CAMERA_BACK),
                                   pull_back=CAMERA_PULL_BACK, pull_back_speed=MAX_SPEED)
        self.platform.setFrustum(fieldOfView=FOV, near=0.3, far=REACH + 150.0)
        self._build_runner_node()
        self.terrain: GeneratedTiles | None = None
        self.sg = sceneGraph(children=[])
        self._start(self.seed)
        for key, method in CONTROLS.items():
            self.addEventHandler('keyboard', name=key, state=1, function=getattr(self, method))
        self.mixer.start_music()
        if self.main_menu:
            self._show_main_menu()
        print('\n'.join(credits.startup(self.woods is not None, self.photos)))

    def _plant_the_forest(self, land: Landscape) -> None:
        """Trees and ground cover from the forest demo's art, where it is installed, for ``land``."""
        art = woods.forest_art() if self.forest else None
        if art is None:
            return
        if self.woods is None:
            self.woods = woods.Woods(woods.species(art), land)
        else:
            self.woods.relocate(land)
        if self.cover is None:
            self.cover = Cover(art, land)
        else:
            self.cover.relocate(land)

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
        self.session = Session(self.land, demo=self.demo)
        self.terrain = GeneratedTiles(
            HillTiles(self.land, self.look, trees=self.woods is None), reach=REACH, fovy=FOV)
        self._clip = ''
        self._game_over_shown = False
        self.hud.reset()
        self._last = systemTime()
        self._look_through(self.session.step(0.0))
        self.terrain.prime(self._eye(), self._viewport_height())
        self._fill_the_cover()
        self.sg.children = [
            Background(skyColor=SKY, skyAngle=[1.0, 1.5], groundColor=[HAZE]),
            Fog(color=HAZE, visibilityRange=HAZE_RANGE),
            DirectionalLight(direction=SUN, color=(1.0, 0.96, 0.88), intensity=2.4, shadows=True),
            self.terrain, self.treasure, self.runner_node,
            *([self.woods, *self.cover.nodes] if self.woods and self.cover else []),
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
        if self.cover is not None:
            self.cover.fill(self._eye())

    def _viewport_height(self) -> float:
        return float(max(1, self.getViewPort()[1]))

    # -- commands ----------------------------------------------------------
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
                on_settings=self._settings, on_credits=self._credits, on_quit=self.OnQuit,
                on_demo=self._demo, demo=self.session.demo))

    def _show_main_menu(self) -> None:
        self.pushOverlay(menu.main_panel(
            on_play=self._resume, on_demo=self._start_demo, on_settings=self._settings,
            on_credits=self._credits, on_quit=self.OnQuit))

    def _credits(self) -> None:
        self.pushOverlay(dialogs.notice(
            'Credits', credits.text(self.woods is not None, self.photos), columns=64))

    def _settings(self) -> None:
        settings.open_settings(self)

    def _resume(self) -> None:
        self._last = systemTime()

    def _demo(self) -> None:
        """Hand the run to the pilot, or take it back, and carry on."""
        self._play_by(not self.session.demo)

    def _start_demo(self) -> None:
        """Hand the run to the pilot, and carry on."""
        self._play_by(True)

    def _play_by(self, pilot: bool) -> None:
        self.session.demo = pilot
        self.mark('demo', playing=pilot)
        self._resume()

    def _restart(self) -> None:
        self.demo = self.session.demo
        self.seed += 1
        self._start(self.seed)

    # -- the frame ------------------------------------------------------------
    def OnIdle(self, *arguments: Any) -> Any:  # noqa: ARG002 OpenGLContext OnIdle callback signature
        """Run the game for the time since the last frame, unless a screen is up, and show it."""
        now = systemTime()
        elapsed = now - self._last
        self._last = now
        paused = self.overlays.visible
        self.mixer.pause(paused)
        if not paused:
            self._apply(self.session.step(elapsed), now)
            if self.session.over and not self._game_over_shown:
                self._game_over_shown = True
                self.pushOverlay(menu.game_over_panel(
                    self.session.game, on_retry=self._restart, on_quit=self.OnQuit))
        self.triggerRedraw(1)
        return 1

    def _apply(self, frame: Frame, now: float) -> None:
        game = self.session.game
        self.mixer.play(frame.sounds)
        pose = frame.pose
        self.model.play(pose.clip, loop=pose.loop, fade=pose.fade, speed=pose.speed,
                        restart=pose.clip != self._clip)
        self._clip = pose.clip
        self.model.update(frame.dt)
        self._place_runner(frame)
        self.treasure.update(game, now)
        self.hud.update(game, frame.events, now)
        self._look_through(frame)
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

    def _place_runner(self, frame: Frame) -> None:
        self.runner_node.translation = frame.position
        self._yaw.rotation = (0.0, 1.0, 0.0, -frame.heading)
        self._pitch.rotation = (1.0, 0.0, 0.0, frame.pitch)
        self._lean.rotation = (0.0, 0.0, 1.0, frame.lean)

    def _look_through(self, frame: Frame) -> None:
        self.camera.offset = np.array(frame.camera_offset)
        self.camera.target(frame.target)
        self.camera.advance(frame.dt, self.session.game.runner.speed)
        self.camera.apply()

    def OnQuit(self, event: Any = None) -> None:
        """Stop the music and the workers, then close the window."""
        self.mixer.stop()
        if self.cover is not None:
            self.cover.shutdown()
        if self.woods is not None:
            self.woods.shutdown()
        if self.terrain is not None:
            self.terrain.shutdown()
        super().OnQuit(event)
