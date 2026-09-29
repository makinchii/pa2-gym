"""Runner for CS272 PA2.

What this file does:
  1. checks that lambda = 0 gives the same q table as normal one-step SARSA
  2. runs the random agent as a baseline
  3. trains SARSA(lambda) for every lambda in LAMBDAS with 5 seeds each
  4. makes the learning curve plot and prints the table
  5. prints one sample episode of the trained greedy policy with the ansi renderer

Run with:  python myrunner.py
"""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import myenv  # registers our environment (only the runner imports this, the agent can't)
from myagent import ACCUMULATING, RandomAgent, SarsaLambdaAgent, argmax_action


# settings
ENV_ID = "cs272/WildsCollector-v0"

LAMBDAS = [0.0, 0.3, 0.6, 0.9, 1.0]
SEEDS = [0, 1, 2, 3, 4]
NUM_EPISODES = 5000

ALPHA = 0.1
GAMMA = 1.0      # no discounting since episodes get cut off at 120 steps anyway
EPS = 0.1
INIT_VAL = 1.0
TRACE = ACCUMULATING

SMOOTH_WINDOW = 100     # how many episodes to average over for the plot
FINAL_EPISODES = 500    # final return = average of the last 500 episodes
THRESHOLD = -80         # target return for the table


def make_agent(env, lam, seed, total_epi=NUM_EPISODES):
    """Makes a SarsaLambdaAgent with the settings above."""
    return SarsaLambdaAgent(
        env,
        gamma=GAMMA,
        alpha=ALPHA,
        eps=EPS,
        lam=lam,
        trace=TRACE,
        total_epi=total_epi,
        init_val=INIT_VAL,
        seed=seed,
    )


def one_step_sarsa(env, num_episodes, seed):
    """Normal one-step SARSA with no traces, only used for the lambda = 0 check."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q = np.full((n_states, n_actions), INIT_VAL, dtype=float)
    rng = np.random.default_rng(seed)

    # picks actions the exact same way as the agent so both see the same episodes
    def choose_action(state):
        if rng.random() < EPS:
            return int(rng.integers(n_actions))
        return argmax_action(q[state], rng)

    for episode in range(num_episodes):
        if episode == 0:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()
        action = choose_action(state)

        while True:
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = choose_action(next_state)

            if terminated:
                target = reward
            else:
                target = reward + GAMMA * q[next_state, next_action]

            # only update the pair we just used
            q[state, action] += ALPHA * (target - q[state, action])

            state = next_state
            action = next_action
            if terminated or truncated:
                break

    return q


def check_lambda_zero(num_episodes=200, seed=0):
    """Checks that SARSA(0) matches one-step SARSA exactly."""
    agent = make_agent(gym.make(ENV_ID), lam=0.0, seed=seed, total_epi=num_episodes)
    agent.learn()

    q_plain = one_step_sarsa(gym.make(ENV_ID), num_episodes, seed)

    biggest_difference = np.max(np.abs(agent.q - q_plain))
    print(f"lambda = 0 check: largest |q difference| after {num_episodes} episodes = {biggest_difference}")
    if np.allclose(agent.q, q_plain):
        print("  PASSED: SARSA(0) matches one-step SARSA\n")
    else:
        print("  FAILED: the trace update is wrong\n")


def moving_average(values, window):
    """Averages over a sliding window, the result is (window - 1) shorter than the input."""
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def first_episode_reaching(curve, threshold, window):
    """Returns the first episode where the smoothed curve hits the threshold, or None if it never does."""
    for i in range(len(curve)):
        if curve[i] >= threshold:
            # curve[i] is the average of episodes i+1 to i+window, so it counts as reached at episode i+window
            return i + window
    return None


def main():
    # 1. lambda = 0 check
    check_lambda_zero()

    # 2. random baseline
    print("Running the random baseline...")
    random_returns = []
    for seed in SEEDS:
        env = gym.make(ENV_ID)
        env.reset(seed=seed)  # RandomAgent doesn't seed the env itself so we do it here
        random_agent = RandomAgent(env, total_epi=NUM_EPISODES, seed=seed)
        random_returns.append(random_agent.learn())
        env.close()
    random_returns = np.array(random_returns)
    random_mean = np.mean(random_returns)
    print(f"  random agent mean return = {random_mean:.1f}\n")

    # 3. lambda sweep, results[lam] has one row per seed and one column per episode
    results = {}
    saved_agent = None
    for lam in LAMBDAS:
        all_seed_returns = []
        for seed in SEEDS:
            print(f"Training lambda = {lam}, seed = {seed}...")
            env = gym.make(ENV_ID)
            agent = make_agent(env, lam=lam, seed=seed)
            all_seed_returns.append(agent.learn())
            env.close()

            # save one trained agent for the sample episode later
            if lam == 0.9 and seed == SEEDS[0]:
                saved_agent = agent
        results[lam] = np.array(all_seed_returns)

    # 4a. learning curve plot
    plt.figure(figsize=(10, 6))
    for lam in LAMBDAS:
        # smooth each seed's curve first, then get the mean and std across seeds
        smoothed = []
        for seed_returns in results[lam]:
            smoothed.append(moving_average(seed_returns, SMOOTH_WINDOW))
        smoothed = np.array(smoothed)
        mean_curve = np.mean(smoothed, axis=0)
        std_curve = np.std(smoothed, axis=0)

        episodes = np.arange(SMOOTH_WINDOW, NUM_EPISODES + 1)
        plt.plot(episodes, mean_curve, label=f"lambda = {lam}")
        plt.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.axhline(random_mean, color="gray", linestyle="--", label="random agent")
    plt.axhline(THRESHOLD, color="black", linestyle=":", label=f"threshold = {THRESHOLD}")
    plt.xlabel("Episode")
    plt.ylabel(f"Return ({SMOOTH_WINDOW}-episode moving average)")
    plt.title(f"SARSA(lambda) on {ENV_ID}  (mean +/- std over {len(SEEDS)} seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("lambda_sweep.png", dpi=150)
    print("\nSaved plot to lambda_sweep.png\n")

    # 4b. results table
    print(f"Threshold: {SMOOTH_WINDOW}-episode moving-average return >= {THRESHOLD}")
    print(f"Final return: mean of the last {FINAL_EPISODES} episodes (mean +/- std across seeds)")
    print(f"Seeds: {SEEDS}\n")
    print(f"{'lambda':>8} | {'episodes to threshold':>22} | {'final return':>16}")
    print("-" * 54)
    for lam in LAMBDAS:
        # find when each seed hits the threshold, then average them
        reach_episodes = []
        for seed_returns in results[lam]:
            curve = moving_average(seed_returns, SMOOTH_WINDOW)
            reach_episodes.append(first_episode_reaching(curve, THRESHOLD, SMOOTH_WINDOW))

        if None in reach_episodes:
            num_reached = len(reach_episodes) - reach_episodes.count(None)
            reach_text = f"{num_reached}/{len(SEEDS)} seeds reached"
        else:
            reach_text = f"{np.mean(reach_episodes):.0f} +/- {np.std(reach_episodes):.0f}"

        final_per_seed = np.mean(results[lam][:, -FINAL_EPISODES:], axis=1)
        final_text = f"{np.mean(final_per_seed):.1f} +/- {np.std(final_per_seed):.1f}"

        print(f"{lam:>8} | {reach_text:>22} | {final_text:>16}")
    print(f"{'random':>8} | {'-':>22} | {random_mean:>16.1f}")

    # 5. sample greedy episode using the ansi renderer
    print("\n===== Sample greedy episode (lambda = 0.9, seed = 0) =====")
    print("Actions: 0 = North, 1 = South, 2 = West, 3 = East\n")
    saved_agent.env = gym.make(ENV_ID, render_mode="ansi")
    episode, reached_terminal = saved_agent.best_run()
    if reached_terminal:
        print(f"Reached the goal after {len(episode)} steps.")
    else:
        print(f"Did not reach the goal; stopped after {len(episode)} steps.")
    print(f"Return = {saved_agent.calc_return(episode)}")


if __name__ == "__main__":
    main()