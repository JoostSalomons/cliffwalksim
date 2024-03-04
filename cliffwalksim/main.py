import gymnasium as gym
import numpy as np

from cliffwalksim.agents.agentfactory import AgentFactory
from cliffwalksim.util.metricstracker import MetricsTracker


def env_interaction(env_str: str, agent_type: str, metrics_tracker: MetricsTracker, time_steps: int = 1000) -> None:
    env = gym.make(env_str, render_mode='human')
    obs, info = env.reset()
    agent = AgentFactory.create_agent(agent_type, env=env)

    for _ in range(time_steps):
        old_obs = obs
        action = agent.policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        print("reward ->", reward)

        agent.update((old_obs, action, reward, obs))

        if terminated or truncated:
            # Episode ended.
            obs, info = env.reset()
            break

    env.close()


if __name__ == "__main__":
    tracker = MetricsTracker()  # For recording returns.
    env_interaction("CliffWalking-v0", 'RANDOM', tracker, 20)
    tracker.plot()
