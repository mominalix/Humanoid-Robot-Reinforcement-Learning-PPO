import pybullet as p
import pybullet_data
import gym
import numpy as np
import tensorflow as tf


# Load the URDF file (Step 2)
def load_urdf(filename):
    p.connect(p.DIRECT)
    p.setPhysicsEngineParameter(fixedTimeStep=0.01)
    p.setDataPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF(filename)
    return robot_id


# Define the Humanoid Robot environment (Step 3)
class HumanoidRobot(gym.Env):
    def __init__(self):
        self.robot_id = load_urdf("humanoid.urdf")
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,)
        )  # 6 joint torques
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(24,)
        )  # Joint positions and velocities

    def step(self, action):
        # Apply action to robot joints (modify based on your URDF file)
        for i in range(6):
            p.setJointMotorControl2(
                self.robot_id,
                i,
                p.CONTROL_MODE_TORQUE,
                targetPosition=0,
                force=action[i],
            )
        p.stepSimulation()

        # Calculate reward (replace with your desired reward function)
        # Here, reward is based on standing upright and moving forward
        base_pos, base_vel = p.getBasePositionAndVelocity(self.robot_id)
        z_pos = base_pos[2]
        forward_vel = base_vel[0]
        reward = 0.1 * z_pos + 2.0 * forward_vel
        done = z_pos < 0.2  # Robot falls

        # Get joint positions and velocities for observation
        joint_states = p.getJointStates(
            self.robot_id, range(p.getNumJoints(self.robot_id))
        )
        observation = np.concatenate(
            [joint_states[i][0] for i in range(len(joint_states))]  # Joint positions
            + [joint_states[i][1] for i in range(len(joint_states))]
        )  # Joint velocities
        return observation, reward, done, {}

    def reset(self):
        p.resetSimulation()
        self.robot_id = load_urdf("humanoid.urdf")
        joint_states = p.getJointStates(
            self.robot_id, range(p.getNumJoints(self.robot_id))
        )
        observation = np.concatenate(
            [joint_states[i][0] for i in range(len(joint_states))]
            + [joint_states[i][1] for i in range(len(joint_states))]
        )
        return observation


# Define the Agent with a simple neural network (Step 4)
class Agent:
    def __init__(self, state_size, action_size):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu", input_shape=(state_size,)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(action_size),
            ]
        )
        self.model.compile(loss="mse", optimizer="adam")

    def act(self, observation):
        action = self.model.predict(observation.reshape(1, -1))[0]
        return np.clip(action, -1, 1)  # Clip action values between -1 and 1


# Training loop (Step 4)
def train(env, agent):
    for generation in range(1000):
        observation = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(observation)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward

        print(f"Generation {generation+1}, Reward: {episode_reward}")

        # Visualize simulation every 10th generation
        if generation % 10 == 0:
            p.connect(p.GUI)
            p.resetSimulation()
            env.reset()
            p.setRealTimeSimulation(True)
