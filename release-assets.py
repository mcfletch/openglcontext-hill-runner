#! /usr/bin/env python3
"""Build the art this game fetches, install it here, and publish it.

The runner, the music and the sounds are 4 MB, which an index should not serve with every release
of the code, so they are a base pack attached to a GitHub release and fetched through
:mod:`OpenGLContext.contentpacks` before the first frame. One command builds it:

    ./release-assets.py                   # build the archive and its registry
    ./release-assets.py --install         # ...and put it in this machine's store
    ./release-assets.py --reinstall       # ...replacing the copy installed there
    ./release-assets.py --write-registry  # ...and write src/openglcontext_hill_runner/packs.json

``--install`` makes a content release testable before it is a release: the game then finds the art in
its own store, as an installed copy would. Nothing here reaches GitHub; ``tools/publish-content.py``
in the workspace attaches the archive to its release from the registry this writes. The archive is
released under its own annotated git tag, ``content-hill-runner-assets-v<N>``, made when the registry
is written and on the commit the archive was built from. A change to the art takes the next number,
since a published file is not replaced: the registry of every earlier release of the code names the
file it was made with.

The registry is written from the archive this built, so its digest and size cannot describe a file
that was never made. The pack holds the art and ``CREDITS.txt``, the terms the art is under, cut from
``ASSET-LICENSES.md``. The options, the install and the push are
:func:`OpenGLContext.contentpacks.publish.main`'s.
"""
from __future__ import annotations

import os

from OpenGLContext import atomicfiles
from OpenGLContext.contentpacks import publish

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "src", "openglcontext_hill_runner", "assets")
CATALOG = os.path.join(HERE, "src", "openglcontext_hill_runner", "packs.json")
LICENSES = os.path.join(HERE, "ASSET-LICENSES.md")
#: The first heading of ASSET-LICENSES.md that is about art the pack does not hold.
NOT_IN_PACK = "## Trees and ground cover"

CREDIT = ("The runner and its animations by Quaternius (CC0 1.0); music by RandomMind, sounds by Kenney, "
          "AntumDeluge and Cici Fyre (all CC0 1.0), and a chime made for this project. Full terms in "
          "CREDITS.txt inside the pack.")


def credits() -> str:
    """The terms of the art the pack holds: ``ASSET-LICENSES.md`` up to the art it does not."""
    with open(LICENSES, encoding="utf-8") as handle:
        text = handle.read()
    return text.split(NOT_IN_PACK)[0].rstrip() + "\n"


def declare(build: publish.Build) -> list[dict]:
    """Build the art pack; what the registry says about it."""
    if not os.path.isdir(ASSETS):
        raise SystemExit(f"no assets at {ASSETS}")
    staged = build.staging("assets")
    for folder, _, files in os.walk(ASSETS):
        for name in files:
            target = os.path.join(staged, os.path.relpath(os.path.join(folder, name), ASSETS))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            atomicfiles.copy_file(os.path.join(folder, name), target)
    atomicfiles.write_bytes(os.path.join(staged, "CREDITS.txt"), credits().encode("utf-8"))
    built = build.archive(staged, "hill-runner-assets")
    return [build.entry(
        "assets", built, title="Hill runner art and sound", copyright=CREDIT, base=True,
        # The first thing the game opens, so its presence is what proves the pack is unpacked.
        marker="runner.glb",
        notes="The runner and its animations, the music and the sounds. Fetched before the first "
              "frame, because the game opens its character before it opens a window.")]


RELEASE = publish.Release(
    namespace="openglcontext-hill-runner",
    url="https://github.com/mcfletch/openglcontext-hill-runner/releases/download/%s/%s",
    catalog=CATALOG, declare=declare,
    into=os.path.join(HERE, "dist", "content"),
    description=__doc__.split("\n\n")[0], title="Hill runner art and sound",
    notes="The art `openglcontext-hill-runner` fetches on its first run. Terms are in the registry "
          "and in CREDITS.txt inside the pack.")


if __name__ == "__main__":
    raise SystemExit(publish.main(RELEASE))
