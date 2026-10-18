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


def test_a_run_from_a_checkout_asks_nothing_about_art_the_checkout_holds(monkeypatch, capsys):
    from openglcontext_hill_runner import run
    monkeypatch.setattr('builtins.input', lambda prompt='': pytest.fail('asked'))
    run.fetch_assets(photos=False)
    assert capsys.readouterr().out == ''


def test_a_run_with_no_art_and_no_yes_stops_without_downloading(tmp_path, monkeypatch):
    from OpenGLContext.contentpacks import Application

    from openglcontext_hill_runner import run
    bare = Application(content.NAMESPACE, content.CATALOG_PATH, base=content.BASE,
                       fallback=str(tmp_path / 'nothing'), root=str(tmp_path / 'store'))
    monkeypatch.setattr(content, 'CONTENT', bare)
    monkeypatch.setattr('builtins.input', lambda prompt='': 'no')
    with pytest.raises(SystemExit, match='Not downloading'):
        run.fetch_assets(forest=False, photos=False)
    assert not (tmp_path / 'store' / content.NAMESPACE / 'assets').exists()


@pytest.fixture
def nothing_here(tmp_path, monkeypatch):
    """Every pack the game runs on missing: no checkout copies and empty stores."""
    from OpenGLContext.contentpacks import Application
    monkeypatch.setenv('OPENGLCONTEXT_CONTENT', str(tmp_path / 'engine'))
    monkeypatch.setattr('OpenGLContext.userpaths.appdatadirectory', lambda: str(tmp_path / 'appdata'))
    bare = Application(content.NAMESPACE, content.CATALOG_PATH, base=content.BASE,
                       fallback=str(tmp_path / 'nothing'), root=str(tmp_path / 'runner'))
    monkeypatch.setattr(content, 'CONTENT', bare)
    monkeypatch.setattr(content, 'FOREST', Application(
        content.FOREST_NAMESPACE, content.FOREST_CATALOG_PATH, base=content.FOREST_BASE,
        root=str(tmp_path / 'forest')))


def test_the_game_runs_on_its_own_art_the_forest_demos_and_the_engines_textures(nothing_here):
    assert [need.pack.key for need in content.required()] == [
        'openglcontext-hill-runner/assets', 'openglcontext-forest/art', 'openglcontext/textures']


def test_the_options_that_do_without_the_forest_and_the_photographs_need_neither(nothing_here):
    assert [need.pack.key for need in content.required(forest=False, photos=False)] == [
        'openglcontext-hill-runner/assets']


def test_the_packs_are_offered_on_one_question(nothing_here, monkeypatch, capsys):
    from openglcontext_hill_runner import run
    questions = []
    monkeypatch.setattr('builtins.input', lambda prompt='': questions.append(prompt) or 'no')
    with pytest.raises(SystemExit, match='Not downloading'):
        run.fetch_assets()
    shown = capsys.readouterr().out
    assert len(questions) == 1
    for title in ('Hill runner art and sound', 'CC0 PBR textures'):
        assert shown.count(title) == 1


def test_the_forest_demos_pack_is_the_one_the_forest_demo_publishes():
    from OpenGLContext.contentpacks import catalog
    spec = importlib.util.find_spec('openglcontext_forest_demo')
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("the forest demo is not installed to compare with")
    published = os.path.join(list(spec.submodule_search_locations)[0], 'packs.json')
    assert catalog.load(content.FOREST_CATALOG_PATH) == catalog.load(published)
