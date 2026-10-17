"""The credits: who made the character, its animations, the sound, the trees and the ground."""
from openglcontext_hill_runner import credits, cues


def test_the_character_and_each_animation_it_plays_are_named():
    lines = credits.startup(forest=False, photos=False)
    text = '\n'.join(lines)
    assert 'Quaternius' in text and 'Universal Animation Library' in text
    for clip in cues.CLIPS:
        assert clip in text


def test_the_slide_is_credited_to_this_project():
    assert any('Slide' in line and 'this project' in line
               for line in credits.startup(forest=False, photos=False))


def test_the_trees_are_credited_only_when_they_are_used():
    without = credits.text(forest=False, photos=False)
    with_trees = credits.text(forest=True, photos=False)
    assert 'Georgeous' not in without
    for author in ('Georgeous', '3D Error 404', 'LOLIPOP'):
        assert author in with_trees
    assert 'CC-BY' in with_trees


def test_the_ground_photographs_are_credited_only_when_they_are_used():
    assert 'ambientCG' not in credits.text(forest=False, photos=False)
    assert 'ambientCG' in credits.text(forest=False, photos=True)


def test_the_music_and_each_sound_are_credited():
    text = credits.text(forest=False, photos=False)
    for name in ('RandomMind', 'Kenney', 'AntumDeluge', 'Cici Fyre'):
        assert name in text


def test_every_credit_says_its_licence():
    for entry in credits.entries(forest=True, photos=True):
        assert entry.licence and entry.author and entry.what
