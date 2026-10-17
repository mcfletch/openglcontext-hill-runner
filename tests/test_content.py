"""Where the game's own art comes from: a pack fetched from a GitHub release, or the checkout's copy."""
import importlib.util
import os
import tarfile

import pytest
import tomllib
from OpenGLContext.contentpacks.application import NotInstalled

from openglcontext_hill_runner import content

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_a_checkout_runs_from_its_own_copy_of_the_art():
    where = content.assets_directory()
    assert os.path.isfile(os.path.join(where, 'runner.glb'))
    assert os.path.isfile(os.path.join(where, 'audio', 'music.ogg'))


def test_the_registry_names_the_art_as_a_pack_on_a_github_release():
    pack = content.CONTENT.base_pack()
    assert pack.key == content.BASE == 'openglcontext-hill-runner/assets'
    assert pack.url.startswith('https://github.com/mcfletch/openglcontext-hill-runner/releases/download/')
    assert len(pack.sha256) == 64 and pack.approximate_bytes > 1_000_000
    assert pack.base and pack.marker == 'runner.glb'


def test_without_the_pack_or_a_checkout_the_art_is_reported_missing(tmp_path, monkeypatch):
    from OpenGLContext.contentpacks import Application
    bare = Application(content.NAMESPACE, content.CATALOG_PATH, base=content.BASE,
                       fallback=str(tmp_path / 'nothing'), root=str(tmp_path / 'store'))
    with pytest.raises(NotInstalled):
        bare.base_directory()


def test_the_wheel_carries_the_code_and_not_the_art():
    with open(os.path.join(HERE, 'pyproject.toml'), 'rb') as handle:
        data = tomllib.load(handle)['tool']['setuptools'].get('package-data', {})
    assert not any('assets' in pattern for patterns in data.values() for pattern in patterns)


def test_the_pack_holds_the_art_and_the_terms_it_is_under(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'release_assets', os.path.join(HERE, 'release-assets.py'))
    release_assets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release_assets)
    into = tmp_path / 'dist'
    assert release_assets.publish.main(release_assets.RELEASE, ['--into', str(into)]) == 0
    with tarfile.open(into / 'hill-runner-assets.tar.gz') as archive:
        names = set(archive.getnames())
        credits = archive.extractfile('CREDITS.txt').read().decode()
    assert {'runner.glb', 'audio/music.ogg', 'audio/coin.ogg', 'CREDITS.txt'} <= names
    assert 'Quaternius' in credits and 'RandomMind' in credits and 'Trees and ground cover' not in credits


def test_a_run_from_a_checkout_asks_nothing(monkeypatch, capsys):
    from openglcontext_hill_runner import run
    monkeypatch.setattr('builtins.input', lambda prompt='': pytest.fail('asked'))
    run.fetch_assets()
    assert capsys.readouterr().out == ''


def test_a_run_with_no_art_and_no_yes_stops_without_downloading(tmp_path, monkeypatch):
    from OpenGLContext.contentpacks import Application

    from openglcontext_hill_runner import run
    bare = Application(content.NAMESPACE, content.CATALOG_PATH, base=content.BASE,
                       fallback=str(tmp_path / 'nothing'), root=str(tmp_path / 'store'))
    monkeypatch.setattr(content, 'CONTENT', bare)
    monkeypatch.setattr('builtins.input', lambda prompt='': 'no')
    with pytest.raises(SystemExit, match='Not downloading'):
        run.fetch_assets()
    assert not (tmp_path / 'store' / content.NAMESPACE / 'assets').exists()
