"""Where the game's own art comes from.

The runner, the music and the sounds are 4 MB that an index should not be asked to serve with every
release of the code, so they are a **base pack**: attached to a GitHub release and fetched through
:mod:`OpenGLContext.contentpacks` before the first frame (:func:`openglcontext_hill_runner.run.fetch_assets`).
A checkout has the same files in ``src/openglcontext_hill_runner/assets`` and runs from them while
the pack is not installed; an installed wheel carries none.

    >>> from openglcontext_hill_runner import content
    >>> content.assets_directory()                         # doctest: +SKIP

:data:`CONTENT` is the engine's :class:`~OpenGLContext.contentpacks.application.Application` for
this game: which registry, which namespace and which pack is the art. Where the art is, is asked
each time it is needed.
"""
import os

from OpenGLContext.contentpacks import Application, ContentStore
from OpenGLContext.contentpacks.application import Need

__all__ = ['BASE', 'CATALOG_PATH', 'CONTENT', 'IN_CHECKOUT', 'NAMESPACE', 'assets_directory', 'required',
           'store']

HERE = os.path.dirname(os.path.abspath(__file__))
#: The registry shipped with the game.
CATALOG_PATH = os.path.join(HERE, 'packs.json')
#: The namespace the game's packs sit under, and the pack it cannot start without.
NAMESPACE = 'openglcontext-hill-runner'
BASE = NAMESPACE + '/assets'
#: The art in a checkout. An installed wheel does not carry it.
IN_CHECKOUT = os.path.join(HERE, 'assets')

CONTENT = Application(NAMESPACE, CATALOG_PATH, base=BASE, fallback=IN_CHECKOUT)


def store(root: str | None = None) -> ContentStore:
    """Where the game's content lives on this machine."""
    return CONTENT.store(root)


def assets_directory(where: ContentStore | None = None) -> str:
    """Where the art is read from, as of now: the base pack once it is installed, else the copy in
    a checkout; :class:`~OpenGLContext.contentpacks.application.NotInstalled` if neither."""
    return CONTENT.base_directory(where)


def required(forest: bool = True, photos: bool = True) -> list[Need]:
    """Every pack the game runs on that is not on this machine yet.

    The game's own art, the forest demo's trees and ground cover (``forest``) and the engine's
    photographic texture pack (``photos``), each as a :class:`~OpenGLContext.contentpacks.application.Need`
    for :func:`~OpenGLContext.contentpacks.application.ensure_needs`, which fetches them on one consent.
    """
    from OpenGLContext.loaders import cc0
    needs = CONTENT.needs()
    if forest:
        from openglcontext_forest_demo import content as forest_content
        needs += forest_content.CONTENT.needs()
    if photos:
        needs += cc0.pack_needs()
    return needs
