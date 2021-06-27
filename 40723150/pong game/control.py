""" Majority of this code was copied directly from Andrej Karpathy's gist:
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# https://raw.githubusercontent.com/omkarv/pong-from-pixels/master/pong-from-pixels.py

 Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gym
import time as t
from gym import wrappers

H = 200 # number of hidden layer neurons
batch_size = 10 # used to perform a RMS prop param update every batch_size steps
learning_rate = 1e-3 # learning rate used in RMS prop
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

resume = 1
render = 1 # render video output

# model initialization
D = 75 * 80 # input dimensionality: 75x80 grid
if resume:
  model = pickle.load(open('pong1.2_save.p', 'rb'))
else:
  print("Resume Error!!")

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
  I = I[35:185] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
  I = I[::2,::2,0] # downsample by factor of 2.
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
  return I.astype(np.float).ravel() # ravel flattens an array and collapses it into a column vector

def policy_forward(x):
  """This is a manual implementation of a forward prop"""
  h = np.dot(model['W1'], x) # (H x D) . (D x 1) = (H x 1) (200 x 1)
  h[h<0] = 0 # ReLU introduces non-linearity
  logp = np.dot(model['W2'], h) # This is a logits function and outputs a decimal.   (1 x H) . (H x 1) = 1 (scalar)
  p = sigmoid(logp)  # squashes output to  between 0 & 1 range
  return p, h # return probability of taking action 2 (UP), and hidden state

env = gym.make("Pong-v0")

observation = env.reset()

prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
  if render: env.render()
  cur_x = prepro(observation) #丟入畫面
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)#兩張畫面相減
  prev_x = cur_x
  aprob, h = policy_forward(x)
  action = 2 if 0.5 < aprob else 3
  xs.append(x) # observation
  hs.append(h) # hidden state
  y = 1 if action == 2 else 0
  observation, reward, done, info = env.step(action)
  reward_sum += reward
  drs.append(reward)

  if done:
    episode_number += 1

    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print ('resetting env. episode reward total was ' + str(reward_sum) + ' running mean: '+ str(running_reward))

    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None

  if reward != 0:
    print (('round %d game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))