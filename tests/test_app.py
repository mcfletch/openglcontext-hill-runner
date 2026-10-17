"""The window: what it is declared to be, and that it draws a frame of the game."""
import os
import subprocess
import sys

import numpy as np
import pytest
from OpenGLContext.context import Context

from openglcontext_hill_runner.app import RunnerContext

pytest.importorskip("glfw")
Image = pytest.importorskip("PIL.Image")


def test_it_is_a_glfw_window_with_no_free_camera():
    assert issubclass(RunnerContext, Context)
    assert RunnerContext.windowSystemName == 'glfw'
    assert not RunnerContext.resolveDefinition().navigation


def test_the_game_binds_the_arrow_keys_and_escape():
    from openglcontext_hill_runner.app import CONTROLS
    assert set(CONTROLS) == {'<left>', '<right>', '<up>', '<down>', '<escape>'}
    for method in CONTROLS.values():
        assert callable(getattr(RunnerContext, method))


def _has_gl():
    return bool(os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY"))


def capture(tmp_path, frames, *arguments):
    env = dict(os.environ)
    env.update(
        OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_DISABLE_FPS_DISPLAY="1",
        OPENGLCONTEXT_HIDDEN="1", OPENGLCONTEXT_AUTO_EXIT_FRAMES=str(frames),
        OPENGLCONTEXT_AUTO_EXIT_CAPTURE_DIR=str(tmp_path),
        OPENGLCONTEXT_AUTO_EXIT_CAPTURE_NAME="shot",
    )
    result = subprocess.run([sys.executable, "-m", "openglcontext_hill_runner",
                             "--no-sound", *arguments],
                            env=env, capture_output=True, text=True, timeout=240)
    shot = tmp_path / "shot.png"
    assert shot.exists(), f"no frame captured; stderr:\n{result.stderr[-2000:]}"
    assert 'Traceback' not in result.stderr, result.stderr[-2000:]
    return np.asarray(Image.open(shot).convert("RGB"), dtype=float)


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_the_window_draws_the_hill_with_sky_ground_and_a_runner(tmp_path):
    pixels = capture(tmp_path, 120, "--demo")
    height = pixels.shape[0]
    ground = pixels[height * 3 // 4:]
    assert pixels.std() > 25.0
    assert (pixels[..., 2] > pixels[..., 0] + 30).mean() > 0.005    # some sky
    assert ground[..., 1].mean() > 40                              # lit ground under the runner
    middle = pixels[height // 3:, pixels.shape[1] // 3: 2 * pixels.shape[1] // 3]
    red = (middle[..., 0] > 140) & (middle[..., 1] < 70) & (middle[..., 2] < 70)
    assert red.sum() > 20                                          # the red bands of her suit


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_the_window_draws_without_the_forest_demos_art_or_any_download(tmp_path):
    pixels = capture(tmp_path, 120, "--demo", "--no-forest", "--no-photos")
    assert pixels.std() > 25.0
    assert (pixels[..., 2] > pixels[..., 0] + 30).mean() > 0.005


RESTARTING = '''
import sys
from openglcontext_hill_runner.app import RunnerContext


class Restarting(RunnerContext):
    frames = 0
    seed = 1
    demo = True
    sound = False
    photos = False
    forest = True

    def OnIdle(self, *arguments):
        type(self).frames += 1
        if self.frames == 20:
            self.first = (self.cover, self.woods, self.terrain)
            self._restart()
        if self.frames == 60:
            now = (self.cover, self.woods, self.terrain)
            kept = [a is b for a, b in zip(self.first, now)]
            print('KEPT', kept, 'SEED', self.seed, flush=True)
        return super().OnIdle(*arguments)


Restarting.ContextMainLoop(size=(640, 360))
'''


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_a_restart_grows_the_same_cover_and_woods_on_the_new_hill_and_draws(tmp_path):
    env = dict(os.environ)
    env.update(OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_DISABLE_FPS_DISPLAY="1",
               OPENGLCONTEXT_HIDDEN="1", OPENGLCONTEXT_AUTO_EXIT_FRAMES="90",
               OPENGLCONTEXT_AUTO_EXIT_CAPTURE_DIR=str(tmp_path),
               OPENGLCONTEXT_AUTO_EXIT_CAPTURE_NAME="shot")
    script = tmp_path / "restart.py"
    script.write_text(RESTARTING)
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                            text=True, timeout=240)
    assert 'Traceback' not in result.stderr, result.stderr[-2000:]
    assert 'KEPT [True, True, False] SEED 2' in result.stdout, result.stdout[-1000:]


TREELESS = '''
import numpy as np
from openglcontext_hill_runner import woods
from openglcontext_hill_runner.app import RunnerContext

woods.chunk_trees = lambda land, index: woods.Trees(
    np.zeros((0, 3), 'f'), np.zeros(0, 'f'), np.zeros(0, 'f'), np.zeros(0, int))


class Treeless(RunnerContext):
    seed = 1
    demo = True
    sound = False
    photos = False
    forest = True


Treeless.ContextMainLoop(size=(640, 360))
'''


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_woods_with_no_trees_in_them_draw_without_error(tmp_path):
    env = dict(os.environ)
    env.update(OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_DISABLE_FPS_DISPLAY="1",
               OPENGLCONTEXT_HIDDEN="1", OPENGLCONTEXT_AUTO_EXIT_FRAMES="40",
               OPENGLCONTEXT_AUTO_EXIT_CAPTURE_DIR=str(tmp_path),
               OPENGLCONTEXT_AUTO_EXIT_CAPTURE_NAME="shot")
    script = tmp_path / "treeless.py"
    script.write_text(TREELESS)
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                            text=True, timeout=240)
    assert (tmp_path / "shot.png").exists(), result.stderr[-2000:]
    assert 'Traceback' not in result.stderr and 'Error' not in result.stderr, result.stderr[-2000:]


F2_PRESS = '''
import os
from OpenGLContext.events.keyboardevents import KeyboardEvent
from openglcontext_hill_runner.app import RunnerContext


class Pressing(RunnerContext):
    seed = 1
    demo = True
    sound = False
    photos = False
    forest = False
    frames = 0

    def screenshotDirectory(self):
        return os.environ['PICTURES']

    def OnIdle(self, *arguments):
        type(self).frames += 1
        if self.frames == 20:
            key = KeyboardEvent()
            key.name, key.state, key.modifiers = '<F2>', 1, (0, 0, 0)
            self.ProcessEvent(key)
        return super().OnIdle(*arguments)


Pressing.ContextMainLoop(size=(640, 360))
'''


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_f2_saves_a_picture_of_the_game_named_for_it(tmp_path):
    pictures = tmp_path / "pictures"
    pictures.mkdir()
    env = dict(os.environ)
    env.update(OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_DISABLE_FPS_DISPLAY="1",
               OPENGLCONTEXT_HIDDEN="1", OPENGLCONTEXT_AUTO_EXIT_FRAMES="50",
               PICTURES=str(pictures))
    script = tmp_path / "press.py"
    script.write_text(F2_PRESS)
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                            text=True, timeout=240)
    assert 'Traceback' not in result.stderr, result.stderr[-2000:]
    saved = sorted(p.name for p in pictures.iterdir())
    assert saved == ['Hill-Runner-0001.png'], saved
    picture = np.asarray(Image.open(pictures / saved[0]).convert("RGB"), dtype=float)
    assert picture.shape[:2] == (360, 640) and picture.std() > 25.0


ALT_F = '''
import os
from OpenGLContext.events.keyboardevents import KeyboardEvent
from openglcontext_hill_runner.app import RunnerContext


class Pressing(RunnerContext):
    seed = 1
    demo = True
    sound = False
    photos = False
    forest = False
    frames = 0

    def OnIdle(self, *arguments):
        type(self).frames += 1
        if self.frames == 5:
            print('START', self._debugOverlay is not None and self._debugOverlay.visible, flush=True)
            self.press()
        if self.frames == 15:
            print('SHOWN', self.debugOverlay.visible, flush=True)
            self.press()
        if self.frames == 25:
            print('HIDDEN', self.debugOverlay.visible, flush=True)
        return super().OnIdle(*arguments)

    def press(self):
        key = KeyboardEvent()
        key.name, key.state, key.modifiers = 'f', 1, (0, 0, 1)
        self.ProcessEvent(key)


Pressing.ContextMainLoop(size=(640, 360))
'''


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_alt_f_shows_and_hides_the_developer_overlay_which_starts_hidden(tmp_path):
    env = dict(os.environ)
    env.update(OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_HIDDEN="1",
               OPENGLCONTEXT_AUTO_EXIT_FRAMES="40")
    env.pop("OPENGLCONTEXT_DISABLE_FPS_DISPLAY", None)
    script = tmp_path / "altf.py"
    script.write_text(ALT_F)
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                            text=True, timeout=240)
    assert 'Traceback' not in result.stderr, result.stderr[-2000:]
    lines = [line for line in result.stdout.splitlines() if line.split(' ')[0] in
             ('START', 'SHOWN', 'HIDDEN')]
    assert lines == ['START False', 'SHOWN True', 'HIDDEN False'], result.stdout[-1000:]


SHADOW_FIT = '''
import numpy as np
from OpenGLContext.passes import shadowmixin as sm
from openglcontext_hill_runner.app import RunnerContext

widest = []
original = sm.ShadowMapMixin._renderLight


def spy(self, *arguments, **named):
    bound = original(self, *arguments, **named)
    if bound is not None:
        widest.append(max(2.0 / abs(float(np.asarray(p)[0, 0])) for _, p in bound['lights']))
    return bound


sm.ShadowMapMixin._renderLight = spy


class Watching(RunnerContext):
    seed = 1
    demo = True
    sound = False
    photos = False
    forest = True
    frames = 0

    def OnIdle(self, *arguments):
        type(self).frames += 1
        if self.frames == 30:
            print('WIDEST', max(widest) if widest else None, flush=True)
        return super().OnIdle(*arguments)


Watching.ContextMainLoop(size=(640, 360))
'''


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_the_suns_shadow_maps_are_fitted_to_the_view_and_not_to_the_forest(tmp_path):
    env = dict(os.environ)
    env.update(OPENGLCONTEXT_RENDERER="pbr", OPENGLCONTEXT_IBL="full", OPENGLCONTEXT_PROFILE="core",
               OPENGLCONTEXT_HIDDEN="1", OPENGLCONTEXT_AUTO_EXIT_FRAMES="40")
    script = tmp_path / "fit.py"
    script.write_text(SHADOW_FIT)
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True,
                            timeout=240)
    assert 'Traceback' not in result.stderr, result.stderr[-2000:]
    line = next(line for line in result.stdout.splitlines() if line.startswith('WIDEST'))
    assert line != 'WIDEST None', 'no shadow map was made'
    assert float(line.split()[1]) < 5000.0, line


def test_shadows_end_inside_the_radius_the_near_trees_are_drawn_to():
    from OpenGLContext.scenegraph.vegetation.field import NEAR_RADIUS
    assert 0.5 * NEAR_RADIUS < RunnerContext.contextDefinition.shadowDistance < NEAR_RADIUS


def test_the_suns_shadows_are_not_black():
    from openglcontext_hill_runner import app
    assert 0.4 <= app.SHADOW_INTENSITY <= 0.8


def test_the_window_asks_for_shadows_with_three_cascades():
    definition = RunnerContext.contextDefinition
    assert definition.shadows is True and definition.shadowCascades == 3


DEMO_FROM_MENU = '''
from openglcontext_hill_runner.app import RunnerContext


class Handing(RunnerContext):
    seed = 2
    demo = False
    sound = False
    photos = False
    forest = False
    frames = 0

    def OnIdle(self, *arguments):
        type(self).frames += 1
        if self.frames == 10:
            print('BEFORE', self.session.demo, flush=True)
            self._demo()
            print('AFTER', self.session.demo, flush=True)
        if self.frames == 150:
            game = self.session.game
            print('LATER', game.state.name, round(game.runner.distance), flush=True)
            self._demo()
            print('BACK', self.session.demo, flush=True)
        return super().OnIdle(*arguments)


Handing.ContextMainLoop(size=(640, 360))
'''


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_the_demo_the_menu_starts_plays_on_and_the_player_can_take_the_run_back(tmp_path):
    env = dict(os.environ)
    env.update(OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_HIDDEN="1",
               OPENGLCONTEXT_AUTO_EXIT_FRAMES="170")
    script = tmp_path / "handing.py"
    script.write_text(DEMO_FROM_MENU)
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                            text=True, timeout=240)
    assert 'Traceback' not in result.stderr, result.stderr[-2000:]
    said = [line for line in result.stdout.splitlines()
            if line.split(' ')[0] in ('BEFORE', 'AFTER', 'LATER', 'BACK')]
    assert [line.split(' ')[0] + ' ' + line.split(' ')[1] for line in said] == [
        'BEFORE False', 'AFTER True', 'LATER RUNNING', 'BACK False'], said


MAIN_SCREEN = '''
from OpenGLContext.ui.widgets import Button
from openglcontext_hill_runner.app import RunnerContext


def button(panel, name):
    if isinstance(panel, Button) and panel.name == name:
        return panel
    for child in getattr(panel, 'children', []) or []:
        found = button(child, name)
        if found is not None:
            return found


class Opening(RunnerContext):
    seed = 2
    demo = False
    sound = False
    photos = False
    forest = False
    main_menu = True
    frames = 0

    def OnIdle(self, *arguments):
        type(self).frames += 1
        game = self.session.game
        if self.frames == 40:
            print('WAITING', self.overlays.visible, self.session.demo, round(game.runner.distance),
                  flush=True)
            demo = button(self.overlays.top, 'demo')
            demo.on_activate(demo)
        if self.frames == 100:
            print('PLAYING', self.overlays.visible, self.session.demo,
                  round(game.runner.distance) > 5, flush=True)
        return super().OnIdle(*arguments)


Opening.ContextMainLoop(size=(640, 360))
'''


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_the_game_opens_on_the_main_screen_and_its_demo_button_starts_the_computer(tmp_path):
    env = dict(os.environ)
    env.update(OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_HIDDEN="1",
               OPENGLCONTEXT_AUTO_EXIT_FRAMES="120")
    script = tmp_path / "opening.py"
    script.write_text(MAIN_SCREEN)
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                            text=True, timeout=240)
    assert 'Traceback' not in result.stderr, result.stderr[-2000:]
    said = [line for line in result.stdout.splitlines()
            if line.split(' ')[0] in ('WAITING', 'PLAYING')]
    assert said == ['WAITING True False 0', 'PLAYING False True True'], said


def test_the_command_opens_on_the_main_screen_unless_the_computer_is_to_play(monkeypatch):
    from openglcontext_hill_runner import run
    played = []
    monkeypatch.setattr(RunnerContext, 'ContextMainLoop', classmethod(lambda cls, **_: played.append(
        (cls.main_menu, cls.demo))))
    monkeypatch.setattr(run.cc0, 'ensure_pack', lambda **_: None)
    for arguments in ([], ['--demo']):
        run.main(arguments + ['--no-photos'])
    assert played == [(True, False), (False, True)]


DAMAGED = '''
from openglcontext_hill_runner.app import RunnerContext


class Damaged(RunnerContext):
    seed = 1
    demo = True
    sound = False
    photos = False
    forest = False
    owed = 0

    def OnInit(self):
        super().OnInit()
        self.session.game.owed = type(self).owed


Damaged.owed = int(__import__('sys').argv[1])
Damaged.ContextMainLoop(size=(640, 360))
'''


def edge_redness(pixels):
    """How much redder than green the outer tenth of the picture is."""
    rim = np.concatenate([pixels[:, :pixels.shape[1] // 10].reshape(-1, 3),
                          pixels[:, -(pixels.shape[1] // 10):].reshape(-1, 3)])
    return float((rim[:, 0] - rim[:, 1]).mean())


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_the_edges_of_the_picture_are_red_while_a_continue_is_paid_for(tmp_path):
    redness = []
    for damage in (0, 100):
        shots = tmp_path / str(damage)
        shots.mkdir()
        env = dict(os.environ)
        env.update(OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_HIDDEN="1",
                   OPENGLCONTEXT_DISABLE_FPS_DISPLAY="1", OPENGLCONTEXT_AUTO_EXIT_FRAMES="60",
                   OPENGLCONTEXT_AUTO_EXIT_CAPTURE_DIR=str(shots),
                   OPENGLCONTEXT_AUTO_EXIT_CAPTURE_NAME="shot")
        script = tmp_path / "damaged.py"
        script.write_text(DAMAGED)
        result = subprocess.run([sys.executable, str(script), str(damage)], env=env,
                                capture_output=True, text=True, timeout=240)
        assert 'Traceback' not in result.stderr, result.stderr[-2000:]
        redness.append(edge_redness(np.asarray(Image.open(shots / "shot.png").convert("RGB"),
                                               dtype=float)))
    assert redness[0] + 4.0 < redness[1], redness


HAZE = '''
from OpenGLContext.scenegraph.basenodes import Fog
from openglcontext_hill_runner.app import REACH, RunnerContext


class Hazy(RunnerContext):
    seed = 1
    demo = True
    sound = False
    photos = False
    forest = False
    frames = 0

    def OnIdle(self, *arguments):
        type(self).frames += 1
        if self.frames == 3:
            fog = [n for n in self.sg.children if isinstance(n, Fog)]
            print('HAZE', len(fog), fog[0].visibilityRange <= REACH, flush=True)
        return super().OnIdle(*arguments)


Hazy.ContextMainLoop(size=(320, 180))
'''


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_the_haze_hides_everything_before_the_edge_of_the_hill_that_is_made(tmp_path):
    env = dict(os.environ)
    env.update(OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_HIDDEN="1",
               OPENGLCONTEXT_AUTO_EXIT_FRAMES="10")
    script = tmp_path / "hazy.py"
    script.write_text(HAZE)
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                            text=True, timeout=240)
    assert 'Traceback' not in result.stderr, result.stderr[-2000:]
    assert 'HAZE 1 True' in result.stdout.splitlines(), result.stdout[-500:]


UNDERGROUND = '''
from OpenGLContext import renderoptions
from openglcontext_hill_runner import atmosphere
from openglcontext_hill_runner.app import RunnerContext


class Under(RunnerContext):
    seed = 1
    demo = True
    sound = False
    photos = False
    forest = False
    frames = 0

    def _start(self, seed):
        super()._start(seed)
        cave = self.land.course.caves_between(0.0, 20000.0)[0]
        self.session.game.runner.distance = (cave.start + cave.end) / 2 - 40.0
        self._breathe()

    def OnIdle(self, *arguments):
        type(self).frames += 1
        if self.frames == 30:
            print('AIR', round(self._fog.visibilityRange), round(self._sun.intensity, 2),
                  round(float(renderoptions.definition(self).iblIntensity), 3),
                  self.session.game.hits, flush=True)
        return super().OnIdle(*arguments)


Under.ContextMainLoop(size=(320, 180))
'''


@pytest.mark.needs_gl
@pytest.mark.skipif(not _has_gl(), reason="no GL display available")
def test_in_a_cave_the_fog_closes_in_and_the_sun_and_the_sky_go_dim(tmp_path):
    env = dict(os.environ)
    env.update(OPENGLCONTEXT_PROFILE="core", OPENGLCONTEXT_HIDDEN="1",
               OPENGLCONTEXT_AUTO_EXIT_FRAMES="40")
    script = tmp_path / "under.py"
    script.write_text(UNDERGROUND)
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                            text=True, timeout=240)
    assert 'Traceback' not in result.stderr, result.stderr[-2000:]
    line = next(line for line in result.stdout.splitlines() if line.startswith('AIR'))
    _, fog, sun, ibl, damage = line.split()
    assert int(fog) == 150 and float(sun) < 0.6 and float(ibl) < 0.1 and int(damage) == 0
