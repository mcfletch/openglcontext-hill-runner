"""The pause and game-over screens."""
import pytest
from OpenGLContext.ui.widgets import Button

from openglcontext_hill_runner import course as c
from openglcontext_hill_runner import game as g
from openglcontext_hill_runner import menu


def buttons(panel):
    found = {}

    def walk(widget):
        if isinstance(widget, Button):
            found[widget.name] = widget
        for child in getattr(widget, 'children', []) or []:
            walk(child)
    walk(panel)
    return found


@pytest.fixture
def actions():
    calls = []
    return calls, {name: (lambda name=name: calls.append(name))
                   for name in ('resume', 'restart', 'settings', 'credits', 'quit')}


def test_the_pause_screen_offers_resume_restart_settings_credits_and_quit(actions):
    calls, handlers = actions
    panel = menu.pause_panel(**{'on_' + k: v for k, v in handlers.items()})
    offered = buttons(panel)
    assert set(offered) == {'resume', 'restart', 'settings', 'credits', 'quit'}
    for button in offered.values():
        button.on_activate(button)
    assert sorted(calls) == ['credits', 'quit', 'restart', 'resume', 'settings']


def test_escape_on_the_pause_screen_resumes(actions):
    calls, handlers = actions
    panel = menu.pause_panel(**{'on_' + k: v for k, v in handlers.items()})
    panel.on_close(panel)
    assert calls == ['resume']


def test_the_pause_screen_is_modal_over_a_dimmed_game(actions):
    _, handlers = actions
    panel = menu.pause_panel(**{'on_' + k: v for k, v in handlers.items()})
    assert panel.modal and panel.scrim


def test_the_game_over_screen_reports_the_run_and_offers_a_retry_or_quit():
    game = g.Game(c.Course(1))
    game.coins, game.points = 9, 21
    game.runner.distance = 412.0
    calls = []
    panel = menu.game_over_panel(game, on_retry=lambda: calls.append('retry'),
                                 on_quit=lambda: calls.append('quit'))
    texts = []

    def collect(widget):
        if hasattr(widget, 'text') and isinstance(widget.text, str):
            texts.append(widget.text)
        for child in getattr(widget, 'children', []) or []:
            collect(child)
    collect(panel)
    joined = ' '.join(texts)
    assert '412' in joined and '9' in joined and '21' in joined
    offered = buttons(panel)
    assert set(offered) == {'retry', 'quit'}
    offered['retry'].on_activate(offered['retry'])
    assert calls == ['retry']


def test_the_game_over_screen_cannot_be_dismissed_without_choosing():
    panel = menu.game_over_panel(g.Game(c.Course(1)), on_retry=lambda: None, on_quit=lambda: None)
    assert not panel.closeOnEscape
