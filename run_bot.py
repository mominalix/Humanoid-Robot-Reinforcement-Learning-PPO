import gym
import pickle
import time

from environment import HumanoidEnv


def visualize_episodes(filename):
    with open(filename, "rb") as f:
        episodes_data = pickle.load(f)

    env = HumanoidEnv()
    env.render()  # Initialize rendering environment

    for idx, episode_data in enumerate(episodes_data, start=1):
        print(f"Visualizing episode {idx}...")
        state = env.reset()

        for step_data in episode_data:
            action = step_data[
                1
            ]  # Extract action (step_data: (state, action, reward, done))
            state, reward, done, _ = env.step(action)
            time.sleep(0.02)  # Adjust delay as needed for visualization speed

            if done:
                break

    env.close()


if __name__ == "__main__":
    visualize_episodes("episodes_data.pkl")
