import gym
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def cartpole_run():
    env=gym.make('CartPole-v1')
    env.reset()
    at=0
    Qt=0

    for i_episode in range(20):
        observation = env.reset()
        for t in range(500):
            env.render()
            #Qt=reward+

            action = at
            #action=np.random.uniform(-1.0,1.0,size=action_size)
            observation, reward, done, info = env.step(action)


            if done:
                print("Episode finished after {} timesteps".format(t+1))
                print("reward : {}".format(reward))
                break
    env.close()

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(1, 5, 2, stride=1)  # [4]
        self.conv2 = nn.Conv1d(5, 10, 2, stride=1)  # [3]
        self.conv3= nn.Conv1d(10,1,2, stride=1)  # [2]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        return torch.round(self.conv3(x))

