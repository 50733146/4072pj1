""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
# need to wait https://github.com/chainer/chainer/issues/8582
import numpy as np
import pickle
import gym

from chainer import cuda
import cupy as cp
import time, threading

#backend
be = cp

# hyperparameters
A = 3   # 2, 3 for no-ops
H = 200 # number of hidden layer neurons
update_freq = 10
batch_size = 1000 # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

device = 1
pickle_name = "pong2_r_save.p"
reward_list = np.array(())
reward_list_name = "pong2_r_rlist.txt"
sum_reward_list = np.array(())
sum_reward_list_name = "pong2_r_Srlist.txt"
running_reward = None
reward_sum = 0
episode_number = 0
resume = 1
render = 0
# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
with cp.cuda.Device(0):
    if resume:
        model = pickle.load(open(pickle_name, 'rb'))
        reward_list = np.loadtxt(reward_list_name)
        sum_reward_list = np.loadtxt(sum_reward_list_name)
        running_reward = sum_reward_list[-1]
        print('resuming')
    else:
        model = {}
        model['W1'] = np.random.randn(D,H) / np.sqrt(D) # "Xavier" initialization
        model['W2'] = np.random.randn(H,A) / np.sqrt(H)
      
    grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
    rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory
def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def softmax(x):
    #if(len(x.shape)==1):
    #  x = x[np.newaxis,...]
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs
  
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    
    return discounted_r

def policy_forward(x):
    if(len(x.shape)==1):
        x = x[np.newaxis,...]

    h = x.dot(model['W1'])
    h[h<0] = 0 # ReLU nonlinearity
    logp = h.dot(model['W2'])
    #p = sigmoid(logp)
    p = softmax(logp)
    return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = eph.T.dot(epdlogp)  
    dh = epdlogp.dot(model['W2'].T)
    dh[eph <= 0] = 0 # backpro prelu

    t = time.time()
    # problem: https://github.com/chainer/chainer/issues/8582
    if(be == cp):
        dh_gpu = cuda.to_gpu(dh, device=0)
        epx_gpu = cuda.to_gpu(epx.T, device=0)
        dW1 = cuda.to_cpu( epx_gpu.dot(dh_gpu) )
    else:
        dW1 = epx.T.dot(dh) 
    

    print((time.time()-t0)*1000, ' ms, @final bprop')

    return {'W1':dW1, 'W2':dW2}



env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]

while True:
    t0  = time.time()
    if render:
        t  = time.time()
        env.render()
        #print((time.time()-t)*1000, ' ms, @rendering')

    t  = time.time()
    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    #print((time.time()-t)*1000, ' ms, @prepo')
  
  
    # forward the policy network and sample an action from the returned probability
    t  = time.time()
    aprob, h = policy_forward(x)
    #print(aprob)
    #action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
    #print((time.time()-t)*1000, ' ms, @forward')

    # roll the dice, in the softmax loss
    u = np.random.uniform()
    aprob_cum = np.cumsum(aprob)
    a = np.where(u <= aprob_cum)[0][0]
    action = a+1
    #if action==4:action = 0
    #print(action)
    #print(u, a, aprob_cum)
  

    # record various intermediates (needed later for backprop)
    t = time.time()
    xs.append(x) # observation
    hs.append(h) # hidden state


    #softmax loss gradient
    dlogsoftmax = aprob.copy()
    dlogsoftmax[0,a] -= 1 #-discounted reward
    dlogps.append(dlogsoftmax)



    # step the environment and get new measurements
    t  = time.time()
    observation, reward, done, info = env.step(action)
    #print(reward, done)
    reward_sum += reward 
    #print((time.time()-t)*1000, ' ms, @env.step')


    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
    #print((time.time()-t0)*1000, ' ms, @whole.step')

    if done: # an episode finished
        #紀錄單局結算獎勵
        reward_list = np.append(reward_list,reward_sum)
        #計算累積局數獎勵平均
        sum_reward = reward_sum if running_reward is None else running_reward
        sum_reward_list = np.append(sum_reward_list, sum_reward)

        episode_number += 1
        t  = time.time()


        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        print(epdlogp.shape)

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % update_freq == 0: #update_freq used to be batch_size
            for k,v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] -= learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        #紀錄reward

        #儲存紀錄和獎勵紀錄
        if episode_number % 100 == 0: pickle.dump(model, open(pickle_name, 'wb'))
        if episode_number % 100 == 0: np.savetxt(reward_list_name, reward_list, fmt="%s")
        if episode_number % 100 == 0: np.savetxt(sum_reward_list_name, sum_reward_list, fmt="%s")
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None
        
        print((time.time()-t)*1000, ' ms, @backprop')


    outstring =""

    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        if reward == -1:
            outstring = ''
        else:
            outstring = '!!!!!!!'
        
        print ('ep '+ str(episode_number) + ': game finished, reward:' +str(reward)+ outstring )