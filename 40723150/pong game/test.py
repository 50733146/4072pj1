import gym
import time
env = gym.make('Pong-v0')
env.reset()
while True:
    env.render()
    time.sleep(0.01)