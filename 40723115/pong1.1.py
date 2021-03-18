""" Trains an agent with (stochastic) Policy Gradients(actor-critic) on Pong. Uses OpenAI Gym. """
# https://raw.githubusercontent.com/schinger/pong_actor-critic/master/pg-pong-ac.py
import numpy as np
#pickle:將對象序列化,以便保存到文件中，並在以後再次加載到程序中
import pickle as pickle
import gym
#copy:複製Python的物件;deepcopy:若該物件有屬性指向其他物件，則也會一併複製
import copy
#read([size]):從當前位置讀取size個位元組，若無引數size，則表示讀取至檔案結束，type為字串
print(open('pong1.1.py').read())
# hyperparameters
H = 200  # number of hidden layer neurons
#batch_size :一次性讀入多少批量的圖片(1次iteration一起訓練batch_size個樣本)
batch_size = 300
#學習率太小，代表對神經網絡進行非常小的權重更新，會使其訓練變非常緩慢；然後學習率太大，可能導致無法收斂
learning_rate = 1e-3
#gamma用於增加好決策機率，降低壞決策機率
gamma = 0.99  # discount factor for reward
#decay_rate :隨時間慢慢減少學習率，意即減慢RMSprop
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
#Momentum :為模擬物理動量的概念，在同方向的維度上學習速度會變快，方向改變的時候學習速度會變慢。
mom_rate = 0.9 #摩擦力;通常設為0.9
#Temporal Difference:通過可調節的步長n來決定向後採樣幾步來更新
td_step = 30  # initial td step
#gamma平方i
gamma_power = [gamma ** i for i in range(td_step + 1)]

shrink_step = True
rmsprop = True
resume = False  # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
if resume:
    model, model_target = pickle.load(open('save.ac', 'rb'))
else:
    model = {}
    #randn函式返回一個或一組樣本，具有標準正態分佈
    #dn表格每個維度
    #返回值為指定維度的array#H為垂直D橫列
    #函式計算給定陣列中每個元素的平方根
    model['W1_policy'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model['b1_policy'] = np.random.randn(H) / np.sqrt(4 * H)
    model['W2_policy'] = np.random.randn(H) / np.sqrt(H)
    model['b2_policy'] = 0.0
    model['W1_value'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model['b1_value'] = np.random.randn(H) / np.sqrt(4 * H)
    model['W2_value'] = np.random.randn(H) / np.sqrt(H)
    model['b2_value'] = 0.0
    model_target = copy.deepcopy(model)
#items() 方法把字典中每对 key 和 value 组成一个元组，并把这些元组放在列表中返回
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update buffers that add up gradients over a batch
momentum = {k: np.zeros_like(v) for k, v in model.items()} #優化器
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    # I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def forward(x, modelType, model=model):
    h = np.dot(model['W1_' + modelType], x) + model['b1_' + modelType]
    #Rectified Linear Unit
#ReLU is not linear. The simple answer is that ReLU 's output is not a straight line,
# it bends at the x-axis. ... In simple terms, linear functions allow you to dissect the feature plane using a straight line.
# But with the non-linearity of ReLU s, you can build arbitrary shaped curves on the feature plane.
    h[h < 0] = 0  # ReLU nonlinearity
    out = np.dot(model['W2_' + modelType], h) + model['b2_' + modelType]
    if modelType == 'policy':
        out = sigmoid(out)
    return out, h


def backward(eph, epx, epd, modelType):
    """ backward pass. (eph is array of intermediate hidden states) """
    #eph是中間隱藏狀態的數組
    #對序列進行求和計算
    db2 = sum(epd)[0]
    dW2 = np.dot(eph.T, epd).ravel()
    dh = np.outer(epd, model['W2_' + modelType])
    dh[eph <= 0] = 0  # backpro prelu
    db1 = sum(dh)
    dW1 = np.dot(dh.T, epx)
    return {'W1_' + modelType: dW1, 'W2_' + modelType: dW2, 'b1_' + modelType: db1, 'b2_' + modelType: db2}


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, h_ps, h_vs, dlogps, vs, tvs, dvs = [], [], [], [], [], [], []
running_reward = None
reward_sum = 0
round_number = 0
while True:
    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h_p = forward(x, 'policy')
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    v, h_v = forward(x, 'value')
    tv, _ = forward(x, 'value', model_target)
    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    h_ps.append(h_p)  # hidden state
    h_vs.append(h_v)
    vs.append(v)
    tvs.append(tv)
    y = 1 if action == 2 else 0  # a "fake label"
    dlogps.append(
        y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    if reward != 0:
        round_number += 1
        if shrink_step and round_number % 10000 == 0:
            if td_step > 15:
                td_step -= 1
        # calcutate td error
        dvs = [0] * len(vs)
        for i in range(len(vs)):
            if len(vs) - 1 - i < td_step:
                dvs[i] = reward * (gamma_power[len(vs) - 1 - i]) - vs[i]
            else:
                dvs[i] = gamma_power[td_step] * tvs[i + td_step] - vs[i]

        # stack together all inputs, hidden states, action gradients, and td for this episode
        epx = np.vstack(xs)
        eph_p = np.vstack(h_ps)
        eph_v = np.vstack(h_vs)
        epdlogp = np.vstack(dlogps)
        epv = np.vstack(dvs)
        xs, h_ps, h_vs, dlogps, vs, tvs, dvs = [], [], [], [], [], [], []  # reset array memory

        # discounted_epv = epv * np.vstack([gamma**i for i in range(len(epv))])
        epdlogp *= epv  # modulate the gradient with advantage (PG magic happens right here.)
        grad_p = backward(eph_p, epx, epdlogp, 'policy')
        grad_v = backward(eph_v, epx, epv, 'value')
        grad = dict(grad_p, **grad_v)

        for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

        if round_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                if rmsprop:
                    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                    momentum[k] = mom_rate * momentum[k] + learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                else:
                    momentum[k] = mom_rate * momentum[k] + learning_rate * g
                model[k] += momentum[k]
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

                if 'value' in k:
                    model_target[k] = mom_rate * model_target[k] + (1 - mom_rate) * model[k]

        print(('round %d game finished, reward: %f' % (round_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
        if round_number % 3000 == 0: pickle.dump((model, model_target), open('save.ac', 'wb'))
    # boring book-keeping
    if done:
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None