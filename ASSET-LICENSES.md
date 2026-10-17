# Asset licenses

Audio under `src/openglcontext_hill_runner/assets/audio/`. Every file is
CC0 1.0 (public domain dedication, <https://creativecommons.org/publicdomain/zero/1.0/>).
Credit is not required; the authors are listed for provenance.

| File | Title | Author | Source page | License | Edit |
| --- | --- | --- | --- | --- | --- |
| music.ogg | Virtual Rush (`virtual_rush_loop_0.mp3`, the looping version) | Vitalezzz | <https://opengameart.org/content/virtual-rush> | CC0 | MP3 converted to Ogg Vorbis (libvorbis q4), metadata stripped |
| footstep.ogg | RPG Audio, `footstep00.ogg` | Kenney (Kenney Vleugels) | <https://kenney.nl/assets/rpg-audio> | CC0 (License.txt in the pack) | Re-encoded to Ogg Vorbis (q5), metadata stripped |
| slide.ogg | Scrapes, `scrape-6.ogg` (object dragged across a cinder block) | AntumDeluge | <https://opengameart.org/content/scrapes> | CC0 (page and LICENSE.txt in the pack) | Trimmed to 1.1 s, 30 ms fade in, 300 ms fade out, +6 dB, re-encoded to Ogg Vorbis (q5) |
| jump_grunt.ogg | RPG Voice Starter Pack, Type 2 (medium-tone female voice), `jump1.wav` | Cici Fyre (cicifyre) | <https://opengameart.org/content/female-rpg-voice-starter-pack> | CC0 | WAV converted to Ogg Vorbis (q5) |
| coin.ogg | RPG Audio, `handleCoins2.ogg` | Kenney (Kenney Vleugels) | <https://kenney.nl/assets/rpg-audio> | CC0 (License.txt in the pack) | Re-encoded to Ogg Vorbis (q5), metadata stripped |
| crash.ogg | Impact Sounds, `impactSoft_heavy_001.ogg` | Kenney | <https://kenney.nl/assets/impact-sounds> | CC0 (License.txt in the pack) | Re-encoded to Ogg Vorbis (q5), metadata stripped |

`slide.ogg` is a scrape across a cinder block.

`chime.ogg` is synthesized for this project from three sine tones with decaying envelopes, and is released under CC0 1.0.

## Character

`src/openglcontext_hill_runner/assets/runner.glb` is the female figure of Quaternius' Universal Base
Characters carrying the Universal Animation Library clips (both CC0 1.0), as prepared in
`grass-clumps/ual/female_character_ual.glb`, with one clip added: `Slide`, a baseball slide authored by
`grass-clumps/slide.py`. The clip is original to this project and is released under the same terms.

## Trees and ground cover

Used when the forest demo's art is installed (`pip install openglcontext-hill-runner[forest]`, then
the demo's content pack); the game reads it from there and ships none of it. The tree models are CC-BY 4.0,
and the game prints these credits when it plants them:

| Model | Author | Source |
| --- | --- | --- |
| Fir tree | Georgeous | <https://skfb.ly/pA8TG> |
| Noel_Pine_Tree | 3D Error 404 | <https://skfb.ly/6XHoJ> |
| Maple trees pack (lowpoly, game ready, LODs) | LOLIPOP | <https://skfb.ly/p9tGx> |

The grass, ferns, nettles, periwinkle and shrub are Poly Haven scans, CC0. The ground and trail photographs
are ambientCG materials (`Grass004` for grass, `Ground042` for the forest floor, `Rock030` for rock,
`Ground080` for sand, `Ground068` for the trails), CC0, fetched by `OpenGLContext.loaders.cc0` on first run
and cached; without a network the game draws procedural surfaces.
