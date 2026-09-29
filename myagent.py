"""Task 2: SARSA(lambda) with eligibility traces.

Do not change the class name or the constructor signature -- the grading harness
constructs this class directly, and it will hand you an environment you have
never seen. Read the sizes off the spaces and never assume anything about what a
state number means. Do not import myenv from this file.
"""

from typing import Any

import numpy as np
import gymnasium as gym

ACCUMULATING = "accumulating"
REPLACING = "replacing"


def argmax_action(values: np.ndarray, rng: np.random.Generator) -> int:
    """Return the index of the largest value, breaking ties uniformly at random.

    Ties are not an edge case here. The table starts uniform, so on the first
    visit to a state every action is tied, and a plain np.argmax would commit
    every state in the table to action 0.

    Args:
        values: the q-values of one state, shape (n_actions,)
        rng: the agent's random generator

    Returns:
        int: an action
    """
    # every action whose value equals the biggest value
    best_actions = np.flatnonzero(values == np.max(values))

    # pick one of them at random (if there is only one, it gets picked)
    return int(rng.choice(best_actions))


class SarsaLambdaAgent:
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        alpha: float = 0.05,
        eps: float = 0.1,
        lam: float = 0.9,
        trace: str = ACCUMULATING,
        total_epi: int = 5_000,
        init_val: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            env: any tabular gym environment. Both spaces are Discrete.
            gamma: discount factor.
            alpha: learning rate.
            eps: exploration rate for a plain (non-decaying) epsilon-greedy.
            lam: the lambda of SARSA(lambda), in [0, 1]. At 0 this must reduce
                to ordinary one-step SARSA.
            trace: "replacing" or "accumulating".
            total_epi: number of training episodes.
            init_val: value every q(s,a) starts at. Setting this at or slightly
                above the best achievable return makes every untried action look
                good, which drives systematic exploration -- on a sparse-reward
                environment that is often what makes learning possible at all.
            seed: seed for the agent's own randomness, for reproducible runs.
        """
        if trace not in (ACCUMULATING, REPLACING):
            raise ValueError(f"unknown trace type: {trace}")

        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.lam = lam
        self.trace = trace
        self.total_epi = total_epi
        self.init_val = init_val
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.q = self.init_qtable(init_val)

    def init_qtable(self, init_val: float = 0.0) -> np.ndarray:
        """Build the q table, shape (n_states, n_actions), filled with init_val."""
        return np.full((self.n_states, self.n_actions), init_val, dtype=float)

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        """Epsilon-greedy action selection over the current q table.

        Args:
            state: the current state
            exploration: explore with probability eps if True; act greedily if
                False. The greedy path is what best_run uses.

        Returns:
            int: an action
        """
        if exploration and self.rng.random() < self.eps:
            # explore: any action with equal probability
            return int(self.rng.integers(self.n_actions))

        # exploit: best action for this state, ties broken at random
        return argmax_action(self.q[state], self.rng)

    def learn(self) -> list[float]:
        """Run SARSA(lambda) for self.total_epi episodes, updating self.q.

        Returns:
            list[float]: the undiscounted return of each training episode, in
            order. myrunner.py plots these.
        """
        returns = []

        for episode in range(self.total_epi):
            # traces are episodic, so they get cleared at the start of every episode
            e = np.zeros((self.n_states, self.n_actions))

            # seed the environment on the first episode, later resets keep using the same random generator, so whole run is reproducible
            if episode == 0:
                state, info = self.env.reset(seed=self.seed)
            else:
                state, info = self.env.reset()

            action = self.eps_greedy(state)
            total = 0.0

            while True:
                next_state, reward, terminated, truncated, info = self.env.step(action)
                total += reward
                next_action = self.eps_greedy(next_state)

                # TD error
                if terminated:
                    # no next state exists, so q(terminal, .) counts as 0
                    delta = reward - self.q[state, action]
                else:
                    # this includes truncated: next_state is a real state
                    delta = reward + self.gamma * self.q[next_state, next_action] - self.q[state, action]

                # bump the trace of the pair we just used
                if self.trace == ACCUMULATING:
                    e[state, action] += 1
                else:
                    e[state, action] = 1

                # for all s in S, a in A from pseudocode, done on the whole table at once: every pair moves by its share of delta, then every trace decays
                self.q += self.alpha * delta * e
                e *= self.gamma * self.lam

                state = next_state
                action = next_action

                if terminated or truncated:
                    break

            returns.append(total)

        return returns

    def best_run(self, max_steps: int = 300) -> tuple[list[tuple[int, int, float]], bool]:
        """Generate one greedy episode under the learned q table, for the report.

        If the environment was made with render_mode="ansi", every step is also
        printed with the environment's renderer.

        Args:
            max_steps: give up after this many steps.

        Returns:
            tuple[
                list[tuple[int,int,float]]: the episode, as [(s, a, r), ...]
                bool: True if it reached a terminal state, False if it ran out
            ]
        """
        episode = []
        reached_terminal = False
        show = self.env.render_mode == "ansi"

        state, info = self.env.reset(seed=self.seed)
        if show:
            print("Start")
            print(self.env.render())
            print()

        for step in range(max_steps):
            action = self.eps_greedy(state, exploration=False)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            episode.append((state, action, reward))

            if show:
                print(f"Step {step + 1}: action = {action}, reward = {reward}")
                print(self.env.render())
                print()

            state = next_state
            if terminated:
                reached_terminal = True
                break
            if truncated:
                break

        return episode, reached_terminal

    def calc_return(self, episode: list[tuple[Any, Any, float]], discounted: bool = False) -> float:
        """Return of an episode given as [(s, a, r), ...]."""
        total = 0.0
        for t, (state, action, reward) in enumerate(episode):
            if discounted:
                total += (self.gamma ** t) * reward
            else:
                total += reward
        return total


class RandomAgent(SarsaLambdaAgent):
    """The baseline your agent has to beat. Already written; do not change it."""

    def learn(self) -> list[float]:
        returns = []
        for _ in range(self.total_epi):
            self.env.reset()
            total = 0.0
            while True:
                action = int(self.rng.integers(self.n_actions))
                _, reward, terminated, truncated, _ = self.env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
        return returns