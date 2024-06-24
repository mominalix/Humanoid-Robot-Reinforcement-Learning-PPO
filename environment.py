# environment.py
import gym
import pybullet as p
import pybullet_data
import numpy as np


class HumanoidEnv(gym.Env):
    def __init__(self):
        super(HumanoidEnv, self).__init__()
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = p.loadURDF("humanoid.urdf", basePosition=[0, 0, 0.8])
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(66,), dtype=np.float32
        )
        self.state = np.zeros(66)
        self.num_joints = p.getNumJoints(self.robot)
        self.episode_number = 0
        self.debug_text_id = None
        self.reset()

    def step(self, action):
        p.setJointMotorControl2(
            self.robot, 0, p.POSITION_CONTROL, targetPosition=action
        )
        p.stepSimulation()
        self.state = self.get_state()
        reward = self.calculate_reward()
        done = self.check_done()
        return np.array(self.state), reward, done, {}

    def reset(self):
        p.resetSimulation()
        self.robot = p.loadURDF("humanoid.urdf", basePosition=[0, 0, 0.8])
        self.state = self.get_state()
        self.update_debug_text()
        return np.array(self.state)

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect(self.physicsClient)

    def get_state(self):
        joint_states = p.getJointStates(self.robot, range(self.num_joints))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        state = joint_positions + joint_velocities
        additional_info = [0.0] * (66 - len(state))  # Ensure state is of length 66
        state += additional_info
        print(f"State dimensions: {len(state)}")  # Debug print
        return np.array(state)

    def calculate_reward(self):
        position, _ = p.getBasePositionAndOrientation(self.robot)
        return position[2]

    def check_done(self):
        position, orientation = p.getBasePositionAndOrientation(self.robot)
        return position[2] < 0.5

    def set_episode_number(self, episode_number):
        self.episode_number = episode_number

    def update_debug_text(self):
        if self.debug_text_id is not None:
            p.removeUserDebugItem(self.debug_text_id)
        self.debug_text_id = p.addUserDebugText(
            f"Episode: {self.episode_number}",
            [0, 0, 1.5],
            textColorRGB=[1, 0, 0],
            textSize=2,
            lifeTime=0,
        )
