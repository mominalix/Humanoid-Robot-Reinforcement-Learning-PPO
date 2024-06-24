import gym
import numpy as np
import torch
import pickle  # For saving episodes

from environment import HumanoidEnv
from agent import PPO


def main():
    env = HumanoidEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPO(state_dim=state_dim, action_dim=action_dim)

    n_episodes = 1000
    save_every = 50  # Save every 50th episode

    episodes_data = []  # List to store episodes data

    for episode in range(n_episodes):
        env.set_episode_number(episode + 1)
        state = env.reset()
        episode_data = []  # List to store data for this episode

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition((state, action, reward, done))
            state = next_state
            episode_data.append((state, action, reward, done))  # Collect episode data

            if done:
                break

        agent.train()
        print(f"Episode {episode + 1} completed")

        if (episode + 1) % save_every == 0:
            episodes_data.append(episode_data)
            print(f"Saving episode {episode + 1} data...")

    # Save all episodes data to a file
    with open("episodes_data.pkl", "wb") as f:
        pickle.dump(episodes_data, f)
    print(f"All episodes data saved to episodes_data.pkl")


if __name__ == "__main__":
    main()
