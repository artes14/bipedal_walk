import gym
import numpy as np


env=gym.make('BipedalWalker-v3')
env.reset()

action_size=env.action_space.shape[0]

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        # action = env.action_space.sample()
        action=np.random.uniform(-1.0,1.0,size=action_size)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))

            break
env.close()


