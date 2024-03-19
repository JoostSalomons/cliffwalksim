import gymnasium as gym

from agents.agentfactory import AgentFactory


def env_interaction(env_str: str, agent_type: str, num_episodes: int = 500) -> None:
    env = gym.make(env_str, render_mode='human')
    obs, info = env.reset()
    agent = AgentFactory.create_agent(agent_type, env=env)

    while True:
        old_obs = obs
        action = agent.policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        agent.update((old_obs, action, reward, obs))

        if terminated or truncated:
            num_episodes -= 1
            # Episode ended.
            obs, info = env.reset()

        if num_episodes == 0:
            break

    env.close()

def sarsa_interaction(env_str: str, agent_type: str, time_steps: int = 1000) -> None:
    env = gym.make(env_str, render_mode='human')
    obs, info = env.reset()
    agent = AgentFactory.create_agent(agent_type, env=env)
    action = agent.policy(obs)
    for _ in range(time_steps):
        old_obs = obs
        obs, reward, terminated, truncated, info = env.step(action)
        next_action = agent.policy(obs)
        agent.update((old_obs, action, reward, obs, next_action))
        action = next_action
        if terminated or truncated:
            num_episodes -= 1
            # Episode ended.
            obs, info = env.reset()

        if num_episodes == 0:
            break

    env.close()

def sarsa_interaction(env_str: str, agent_type: str, time_steps: int = 1000) -> None:
    env = gym.make(env_str, render_mode='human')
    obs, info = env.reset()
    agent = AgentFactory.create_agent(agent_type, env=env)
    action = agent.policy(obs)
    for _ in range(time_steps):
        old_obs = obs
        obs, reward, terminated, truncated, info = env.step(action)
        next_action = agent.policy(obs)
        agent.update((old_obs, action, reward, obs, next_action))
        action = next_action
        if terminated or truncated:
            # Episode ended.
            obs, info = env.reset()
            break

    env.close()


if __name__ == "__main__":
    # env_interaction("CliffWalking-v0", 'RANDOM', 20)
    sarsa_interaction("CliffWalking-v0", 'SARSA', 500)
