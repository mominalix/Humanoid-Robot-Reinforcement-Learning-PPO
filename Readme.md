# Humanoid Robot Reinforcement Learning

This repository contains a project that utilizes a reinforcement learning algorithm to make a humanoid robot walk in a PyBullet simulation. The project is structured into multiple Python files and uses a provided URDF file for the humanoid robot. The training process prints the reward for every generation and performs a PyBullet simulation of every Nth generation. The simulation animates the model and waits until it falls before it ends.

![simulation video](Simulation.gif)

## Project Structure

- `train.py`: Main script to train the humanoid robot.
- `agent.py`: Defines the PPO agent used for training.
- `environment.py`: Custom Gym environment for the humanoid robot.
- `humanoid.urdf`: URDF file for the humanoid robot.

## Requirements

- Python 3.7+
- PyBullet
- Gym
- NumPy
- PyTorch

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Humanoid-Robot-Reinforcement-Learning-PPO.git
cd Humanoid-Robot-Reinforcement-Learning-PPO
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not provided, manually install the packages:

```bash
pip install pybullet gym numpy torch
```

## Usage

### Training the Humanoid Robot

To train the humanoid robot, run the `train.py` script:

```bash
python train.py
```

This script will:

1. Initialize the custom Gym environment (`HumanoidEnv`).
2. Initialize the PPO agent.
3. Train the agent for a specified number of episodes.
4. Print the reward for every episode.
5. Visualize the robot's behavior every Nth episode.

### Visualizing Saved Episodes

To visualize a saved episode, you can use the `visualize_episode` function in the `train.py` script. This function is called automatically every Nth episode during training. If you want to visualize a specific saved model or episode, modify the `visualize_episode` function as needed.

### Customizing Parameters

You can customize various parameters in the `train.py` script, such as the number of episodes and the frequency of visualization. Here are some key variables you might want to adjust:

- `n_episodes`: Number of episodes for training.
- `visualize_every`: Frequency of visualization (e.g., every 100 episodes).

```python
n_episodes = 1000
visualize_every = 100
```

## Code Explanation

### `environment.py`

Defines the `HumanoidEnv` class, a custom Gym environment for the humanoid robot:

- `__init__`: Initializes the environment and loads the URDF file.
- `step`: Executes a step in the environment using the given action.
- `reset`: Resets the environment to its initial state.
- `render`: Renders the environment (not used in this example).
- `close`: Closes the environment.
- `get_state`: Retrieves the current state of the robot.
- `calculate_reward`: Calculates the reward based on the robot's position.
- `check_done`: Checks if the episode is done.
- `set_episode_number` and `update_debug_text`: Used for displaying the current episode number in the PyBullet GUI.

### `agent.py`

Defines the PPO agent and the Actor-Critic neural network:

- `ActorCritic`: Neural network with separate actor and critic components.
- `PPO`: Proximal Policy Optimization agent that interacts with the environment, stores transitions, and performs training.

### `train.py`

Script for training and visualizing the humanoid robot:

- `main`: Initializes the environment and agent, runs the training loop, and handles visualization.
- `visualize_episode`: Visualizes the robot's behavior for a specific episode.

### `main.py`

Main script for running saved episodes from episodes_data.pkl

## Troubleshooting

If you encounter issues during training or visualization, ensure that:

1. All required packages are installed.
2. The URDF file (`humanoid.urdf`) is in the correct location.
3. You are using a compatible version of Python (3.7+).

For further assistance, feel free to open an issue on the repository.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.
