"""An environment based off of Pokemon for CS272 PA2."""

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import registry, register


# Defines constants used in the program.
GRID_SIZE = 6
SCORE_VALUES = 12
TARGET_SCORE = 12

# Rewards terminating the episode before truncation.
TERMINAL_SUCCESS_REWARD = 75

# Coordinates are in (row, column) order. The agent begins in the upper-left corner on Plains at (0,0).
TERRAIN_MAP = (
    ("P", "P", "P", "P", "G", "G"),
    ("P", "P", "P", "P", "G", "F"),
    ("P", "P", "P", "G", "F", "F"),
    ("P", "P", "G", "F", "F", "R"),
    ("P", "G", "F", "F", "R", "X"),
    ("P", "P", "F", "R", "X", "X"),
)

TERRAIN_NAMES = {
    "P": "Plains",
    "G": "Grass",
    "F": "Forest",
    "R": "Rocks",
    "X": "Ruins",
}

# Different terrains have different risk/reward ratios.
TERRAIN_MOVEMENT_REWARDS = {
    "P": -1,
    "G": -1,
    "F": -2,
    "R": -3,
    "X": -4,
}

# Each terrain type has a different distribution of creature rarities.
ENCOUNTER_OUTCOMES = ("None", "Common", "Uncommon", "Rare", "Legendary")

# The order of the encounter outcomes is the same as their order below. For example, Plains has a 100% chance to encounter nothing.
TERRAIN_ENCOUNTER_PROBABILITIES = {
    "P": (1.00, 0.00, 0.00, 0.00, 0.00),
    "G": (0.92, 0.06, 0.01, 0.01, 0.00),
    "F": (0.88, 0.06, 0.04, 0.01, 0.01),
    "R": (0.86, 0.05, 0.05, 0.03, 0.01),
    "X": (0.88, 0.02, 0.03, 0.05, 0.02),
}

# Gives different rewards based on the rarity. The reward assesses the performance of the agent.
CREATURE_REWARDS = {
    "None": 0,
    "Common": 2,
    "Uncommon": 5,
    "Rare": 10,
    "Legendary": 20,
}

# Points is the termination condition. An agent needs 12 points to successfully terminate the episode.
CREATURE_COLLECTION_POINTS = {
    "None": 0,
    "Common": 1,
    "Uncommon": 1,
    "Rare": 2,
    "Legendary": 4,
}

# Defines the action space. Remember that coordinates are represented as (row, column).
ACTION_DELTAS = {
    0: (-1, 0),  # north
    1: (1, 0),   # south
    2: (0, -1),  # west
    3: (0, 1),   # east
}


class WildsCollectorEnv(gym.Env):
    """Navigate a wilderness and collect 12 points of creatures."""
    
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.observation_space = spaces.Discrete(GRID_SIZE * GRID_SIZE * SCORE_VALUES) # 6 * 6 * 12 = 432 states
        self.action_space = spaces.Discrete(len(ACTION_DELTAS)) # 4 possible actions
        self.row = 0
        self.column = 0
        self.collection_score = 0

    def reset(self, *, seed=None, options=None):
        """Reset to the northwest Plains tile with no collection points."""
        super().reset(seed=seed)
        self.row = 0
        self.column = 0
        self.collection_score = 0
        observation = self._encode_state(self.row, self.column, self.collection_score) # the intial state is 0, score = (row * 6 + column) * 12 + collection_score
        info = self._build_info(encounter=None, valid_move=None)
        return observation, info

    def step(self, action):
        """Move one tile, sample its encounter, and return the resulting transition."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected an integer from 0 to 3.")

        row_delta, column_delta = ACTION_DELTAS[action]

        # calculates the next position
        next_row = self.row + row_delta
        next_column = self.column + column_delta

        if not self._is_in_bounds(next_row, next_column):
            # Boundary collisions do not enter a terrain tile, so no encounter is sampled.
            observation = self._encode_state(self.row, self.column, self.collection_score)
            info = self._build_info(encounter=None, valid_move=False)
            return observation, -1, False, False, info

        # moves to a valid tile
        self.row = next_row
        self.column = next_column

        # determines what happens at this state
        terrain_type = self._get_terrain(self.row, self.column)
        movement_reward = TERRAIN_MOVEMENT_REWARDS[terrain_type]
        encounter = self._sample_encounter(terrain_type)
        creature_reward = CREATURE_REWARDS[encounter]
        self.collection_score += CREATURE_COLLECTION_POINTS[encounter]
        reward = movement_reward + creature_reward
        terminated = self.collection_score >= TARGET_SCORE

        if terminated:
            reward += TERMINAL_SUCCESS_REWARD
            # Score 12 is terminal but outside Discrete(432), whose score component ends at 11.
            observation = self._encode_state(self.row, self.column, SCORE_VALUES - 1)
        else:
            observation = self._encode_state(self.row, self.column, self.collection_score)

        info = self._build_info(encounter=encounter, valid_move=True)

        # The TimeLimit wrapper sets the truncated flag, not the class itself.
        return observation, reward, terminated, False, info

    def render(self):
        """Return an ANSI-friendly text view of the map and current environment state."""
        if self.render_mode != "ansi":
            return None

        map_rows = []

        # loops thru each map row
        for row_index, terrain_row in enumerate(TERRAIN_MAP):
            rendered_tiles = []

            # for each terrain tile in a row
            for column_index, terrain_symbol in enumerate(terrain_row):

                # checks if the agent is currently there
                if (row_index, column_index) == (self.row, self.column):
                    rendered_tiles.append(f"[{terrain_symbol}]") # agent's location is represented by [ ] around a terrain symbol, e.g. [P].
                else:
                    rendered_tiles.append(f" {terrain_symbol} ") # Otherwise, render the plain symbol, e.g. P.
            map_rows.append(" ".join(rendered_tiles))

        # Looks up the current terrain
        terrain_type = self._get_terrain(self.row, self.column)

        # shows stats about current run.
        details = [
            "\n".join(map_rows),
            f"Collection score: {self.collection_score} / {TARGET_SCORE}",
            f"Position: ({self.row}, {self.column})",
            f"Terrain: {TERRAIN_NAMES[terrain_type]}",
            "",
            "P = Plains   G = Grass   F = Forest",
            "R = Rocks    X = Ruins   [ ] = Agent",
        ]
        return "\n".join(details)

    def close(self):
        """Release resources; this text-only environment has none to release."""
        pass

    def _encode_state(self, row, column, collection_score):
        """Encode each nonterminal position-score pair into one unique discrete state."""

        # (row, column) becomes a number between 0 to 35.
        position_index = row * GRID_SIZE + column

        # state value is position * 12 + the current collection score, from 0 to 431.
        return position_index * SCORE_VALUES + collection_score

    def _decode_state(self, state):
        """Decode a valid nonterminal observation into row, column, and score."""
        position_index, collection_score = divmod(state, SCORE_VALUES)
        row, column = divmod(position_index, GRID_SIZE)
        return row, column, collection_score

    def _get_terrain(self, row, column):
        """Return the terrain symbol at a valid map coordinate."""
        return TERRAIN_MAP[row][column]

    def _is_in_bounds(self, row, column):
        """Return whether a coordinate is inside the fixed wilderness map."""
        return 0 <= row < GRID_SIZE and 0 <= column < GRID_SIZE

    def _sample_encounter(self, terrain_type):
        """Sample one terrain-dependent outcome using Gymnasium's seeded generator."""
        # self.np_random is initialized by reset, so same seed and actions reproduce draws.
        probabilities = TERRAIN_ENCOUNTER_PROBABILITIES[terrain_type]
        outcome_index = self.np_random.choice(len(ENCOUNTER_OUTCOMES), p=probabilities)
        return ENCOUNTER_OUTCOMES[outcome_index]

    def _build_info(self, encounter, valid_move):
        """Build diagnostic information without adding hidden learning state."""
        terrain_type = self._get_terrain(self.row, self.column)
        return {
            "position": (self.row, self.column),
            "terrain": TERRAIN_NAMES[terrain_type],
            "collection_score": self.collection_score,
            "encounter": encounter,
            "valid_move": valid_move,
        }


# Avoid errors when a notebook or test runner imports this module more than once.
if "cs272/WildsCollector-v0" not in registry:
    register(
        id="cs272/WildsCollector-v0",
        entry_point="myenv:WildsCollectorEnv",
        max_episode_steps=120,
    )
