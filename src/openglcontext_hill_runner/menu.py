"""The screens over the game: pause, and the end of a run, from the engine's overlay UI."""
from collections.abc import Callable
from gettext import gettext as _

from OpenGLContext.ui.layout import Column
from OpenGLContext.ui.panel import Panel
from OpenGLContext.ui.widgets import PRIMARY, Button, Label

from openglcontext_hill_runner.game import Game

Action = Callable[[], None]


def _button(text: str, name: str, action: Action, panel: Panel, chosen: list[str],
            role: str = '') -> Button:
    """A button that closes the panel and then does ``action``.

    ``chosen`` records that a button did it, so the panel's own close handler,
    which stands for dismissing it some other way, does not also run.
    """
    button = Button(text=text, name=name)
    if role:
        button.role = role

    def activate(widget: object) -> None:  # noqa: ARG001 the engine calls a button with itself
        chosen.append(name)
        panel.close()
        action()
    button.on_activate = activate
    return button


def pause_panel(on_resume: Action, on_restart: Action, on_settings: Action,
                on_credits: Action, on_quit: Action) -> Panel:
    """Escape's screen. Closing it any other way than a button resumes."""
    panel = Panel(title=_('Paused'), modal=True, scrim=True, preferredColumns=28)
    chosen: list[str] = []
    panel.children = [Column(spacing=6, children=[
        _button(_('Resume'), 'resume', on_resume, panel, chosen, PRIMARY),
        _button(_('Restart'), 'restart', on_restart, panel, chosen),
        _button(_('Settings'), 'settings', on_settings, panel, chosen),
        _button(_('Credits'), 'credits', on_credits, panel, chosen),
        _button(_('Quit'), 'quit', on_quit, panel, chosen),
    ])]
    panel.on_close = lambda _closing: None if chosen else on_resume()
    return panel


def game_over_panel(game: Game, on_retry: Action, on_quit: Action) -> Panel:
    """The end of a run: how far, and what was collected."""
    panel = Panel(title=_('Run over'), modal=True, scrim=True, closeOnEscape=False,
                  preferredColumns=30)
    chosen: list[str] = []
    panel.children = [Column(spacing=6, children=[
        Label(text=_('You ran %d m') % game.meters, name='distance'),
        Label(text=_('%d coins, %d points') % (game.coins, game.points), name='treasure'),
        _button(_('Run again'), 'retry', on_retry, panel, chosen, PRIMARY),
        _button(_('Quit'), 'quit', on_quit, panel, chosen),
    ])]
    return panel
