"""Who made what the game shows and plays: the credits window and the lines printed at start.

The character and its animations come first, since they are what the player watches. What
is listed depends on what the game is using: the trees and ground cover are the forest
demo's art and appear only where it is installed, and the ground photographs only where
they were fetched.
"""
from dataclasses import dataclass

from openglcontext_hill_runner import cues


@dataclass(frozen=True)
class Credit:
    """One credit: what it is for, who made it, the licence it is under and where it is from."""

    what: str
    author: str
    licence: str
    source: str


CHARACTER = Credit('Runner: the female figure of the Universal Base Characters',
                   'Quaternius', 'CC0 1.0', 'https://quaternius.com')
ANIMATIONS = Credit('Animations %s: the Universal Animation Library' % ', '.join(
    clip for clip in cues.CLIPS if clip != 'Slide'),
    'Quaternius', 'CC0 1.0', 'https://quaternius.com')
SLIDE = Credit('Animation Slide: a baseball slide added to the figure (grass-clumps/slide.py)',
               'this project', 'CC0 1.0', 'hill-runner')

AUDIO = (
    Credit('Music: Virtual Rush', 'Vitalezzz', 'CC0 1.0',
           'https://opengameart.org/content/virtual-rush'),
    Credit('Footsteps, coins: RPG Audio; impacts: Impact Sounds', 'Kenney', 'CC0 1.0',
           'https://kenney.nl'),
    Credit('Slide: Scrapes', 'AntumDeluge', 'CC0 1.0',
           'https://opengameart.org/content/scrapes'),
    Credit('Jump grunt: RPG Voice Starter Pack', 'Cici Fyre', 'CC0 1.0',
           'https://opengameart.org/content/female-rpg-voice-starter-pack'),
    Credit('Jewel chime', 'this project', 'CC0 1.0', 'hill-runner'),
)

TREES = (
    Credit('Tree: Fir tree', 'Georgeous', 'CC-BY 4.0', 'https://skfb.ly/pA8TG'),
    Credit('Tree: Noel_Pine_Tree', '3D Error 404', 'CC-BY 4.0', 'https://skfb.ly/6XHoJ'),
    Credit('Trees: Maple trees pack (lowpoly, game ready, LODs)', 'LOLIPOP', 'CC-BY 4.0',
           'https://skfb.ly/p9tGx'),
    Credit('Grass, ferns, nettles, periwinkle and shrub', 'Poly Haven', 'CC0 1.0',
           'https://polyhaven.com'),
)

GROUND = Credit('Ground and trail photographs (Grass004, Ground068 and others)', 'ambientCG',
                'CC0 1.0', 'https://ambientcg.com')


def entries(forest: bool, photos: bool) -> list[Credit]:
    """The credits for what the game is using."""
    return [CHARACTER, ANIMATIONS, SLIDE, *AUDIO, *(TREES if forest else ()),
            *((GROUND,) if photos else ())]


def text(forest: bool, photos: bool) -> str:
    """The credits window's text: one block for each credit."""
    return '\n\n'.join('%s\n  by %s, %s\n  %s' % (e.what, e.author, e.licence, e.source)
                       for e in entries(forest, photos))


def startup(forest: bool, photos: bool) -> list[str]:
    """The lines printed as the game starts: the character and its animations, then the rest."""
    lines = ['%s, by %s (%s)' % (e.what, e.author, e.licence)
             for e in entries(forest, photos)]
    return ['Credits:'] + ['  ' + line for line in lines]
