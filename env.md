# Wilds Collector Environment

**Registered ID:** `cs272/WildsCollector-v0`

## Description

`WildsCollectorEnv` is a creature-collection wilderness environment. The
agent travels through a fixed map, automatically collecting any creature that
appears after a valid move. The objective is to reach at least 12 collection
points. Terrain affects both the movement reward and the probability of each
possible encounter.

## Constructor

Create the environment with `WildsCollectorEnv(render_mode=None)`, or through
Gymnasium with:

```python
gym.make("cs272/WildsCollector-v0", render_mode="ansi")
```

`render_mode="ansi"` enables `render()` to return a human-readable string.

## Map

The 6x6 map is fixed. Coordinates use zero-based `(row, column)` positions.
The agent starts at `(0, 0)` on Plains with collection score 0.

```
P P P P G G
P P P P G F
P P P G F F
P P G F F R
P G F F R X
P P F R X X
```

`P` is Plains, `G` is Grass, `F` is Forest, `R` is Rocks, and `X` is Ruins.

## Observation Space

The observation space is `Discrete(432)`. A nonterminal state contains exactly
the agent position and collection score. There are 36 positions and the 12
directly encoded nonterminal scores 0 through 11.

```
position_index = row * 6 + column
state = position_index * 12 + collection_score
```

To decode a state:

```
position_index, collection_score = divmod(state, 12)
row, column = divmod(position_index, 6)
```

When the true collection score reaches 12 or more, the episode terminates.
That score is outside the observation space, so the terminal observation uses
the current position encoded with score 11. The true score remains available
in `info`, and a learning agent must not bootstrap from a terminal transition.

## Action Space

The action space is `Discrete(4)`:

| Action | Direction | Row/column change |
| --- | --- | --- |
| 0 | North | `(-1, 0)` |
| 1 | South | `(1, 0)` |
| 2 | West | `(0, -1)` |
| 3 | East | `(0, 1)` |

## Transition Dynamics

For a valid neighboring move, the agent enters the destination terrain, pays
that terrain's movement reward, and then samples exactly one encounter using
the destination terrain's probability distribution. Any encountered creature
is collected automatically, adding its collection points to the persistent
collection score.

The diagnostic `info` dictionary contains `position`, `terrain`,
`collection_score`, `encounter`, and `valid_move`.

### Invalid Movement

A move outside the map boundary leaves the agent in place. It samples no
encounter, adds no collection points, returns reward `-1`, and sets
`info["valid_move"]` to `False`.

## Rewards

For a valid move that does not end the episode, the reward is:

```
movement reward + creature reward
```

If the move raises the collection score to at least 12, the final transition
also receives a `+75` terminal-success reward.

## Terrain Rules

| Terrain | Movement reward | No encounter | Common | Uncommon | Rare | Legendary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plains | -1 | 100% | 0% | 0% | 0% | 0% |
| Grass | -1 | 92% | 6% | 1% | 1% | 0% |
| Forest | -2 | 88% | 6% | 4% | 1% | 1% |
| Rocks | -3 | 86% | 5% | 5% | 3% | 1% |
| Ruins | -4 | 88% | 2% | 3% | 5% | 2% |

## Creature Outcomes

| Encounter outcome | Added reward | Collection points |
| --- | ---: | ---: |
| No encounter | 0 | 0 |
| Common | +2 | +1 |
| Uncommon | +5 | +1 |
| Rare | +10 | +2 |
| Legendary | +20 | +4 |

## Episode Start

Each episode starts with the agent at `(0, 0)` on Plains and collection score
0. Calling `reset(seed=k)` initializes the environment's random number
generator with seed `k`.

## Episode End

An episode terminates successfully when the collection score reaches at least
12. The final transition receives the terminal-success reward described above.

An episode truncates after 120 steps if it has not already terminated.

## Randomness and Seeding

Encounters are stochastic and depend on destination terrain. Every random draw
uses Gymnasium's `self.np_random` generator, initialized during `reset()`. Two
fresh environments reset with the same seed and given the same action sequence
produce the same observations, rewards, encounters, collection scores, and
termination flags.

## ANSI Rendering

With `render_mode="ansi"`, `render()` returns the full map as a string. The
current terrain is bracketed, such as `[F]` for an agent on Forest, while other
tiles retain aligned spacing. The rendering also includes collection score and
target, zero-based position, current terrain name, and a legend for every
terrain symbol and the bracketed agent tile.
