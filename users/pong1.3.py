"""
First test using open AI Gym environment.
In order to keep it 'simple' I decided to start with a genetic/evolutionary algorithm which tries to learn how to play pong against a computer opponent.
After 30 generations (with 12 individuals in each generation) some modest learning is indeed shown but the opponent is still superior.
This is not a general algorithm.
Big thanks to OpenAI for providing this cool environment!
Thanks also to this post for inspiration:
https://becominghuman.ai/genetic-algorithm-for-reinforcement-learning-a38a5612c4dc
/H.Roos
"""

import numpy as np
import gym
import time
import random
from scipy.ndimage.morphology import binary_dilation
from gym import wrappers

chromosone_colums = 5  # 3 for ball in direction up,down or straight + one for cursor + one for no ball
genes_per_column=40
chromosone_len = genes_per_column * chromosone_colums
chromosone_max_val = 80
ball_colour = 236

# ball colour = (236,236,236)
# enenmy colour = (213,130,74)
# own colour = (92,186,92)
# background = (144,72,17)


def mutation(policy, p=0.05):
    new_policy = policy.copy()
    for i in range(chromosone_len):
        rand = np.random.uniform()
        if rand < p:
            #print(new_policy[i])
            new_policy[i] = np.random.choice(chromosone_max_val)
    return new_policy
  
def crossover(policy1, policy2):
    new_policy = policy1.copy()
    for i in range(chromosone_len):
        rand = np.random.uniform()
        if rand > 0.5:
            new_policy[i] = policy2[i]
    return new_policy
  
def gen_random_policy():
    return np.random.choice(chromosone_max_val, size=((chromosone_len)))
  
def action_out(policy,ball_dir,obs):
    tot = 0
    obs=obs.flatten()
    if ball_dir is None: # no ball is visible
        cursor_gen_offset = 2*genes_per_column
        for n in range(len(obs)):
            tot += (policy[n+cursor_gen_offset]*obs[n] - (obs[n]*chromosone_max_val)/2)
    else:
        cursor_gen_offset = 3*genes_per_column
        if (ball_dir[1]>0):
            ball_col = 1
        elif (ball_dir[1]==0):
            ball_col = 2
        else:
            ball_col = 3
        for n in range(len(obs)):
            if (obs[n] > 182): # neg number, ball
                tot -= policy[n+ball_col*genes_per_column]*4 # using 4 as weight
            else:
                tot += policy[n+cursor_gen_offset]*obs[n]
    if (tot > 0):
        return 2 
    else:
        return 3
       
def run_episode(env, policy, episode_len=100000, render=False):
    total_reward = 0
    ball_dir = (0,0)
    prev_obs = None
    obs = env.reset()
    episodes = 0
 
    for t in range(episode_len*1000): #1000 is a workaround when running Monitor Wrapper
        if render:
            env.render()
            #time.sleep(0.1)
        ball_dir,prev_obs = detect_ball_dir(ball_dir,obs,prev_obs)
        action = action_out(policy,ball_dir,process_image(obs))
        obs, reward, done, _ = env.step(action)
        episodes += 1
        if (episodes < episode_len): # workaround when running Monitor
            total_reward += reward   # only count reward until max no of episodes
        if done:
            #print("------END------")
            break
    return total_reward

def detect_ball_dir(ball_dir,obs,prev_obs):
    obs = obs[:,:,0]   # remove the second and third RGB value
    obs = obs[34:194] #crop image
    if prev_obs is not None:
        current_pos = np.argwhere(obs==ball_colour)
        prev_pos = np.argwhere(prev_obs==ball_colour)
        if (len(current_pos)>0 and len(prev_pos)>0):
            diff= current_pos[0] - prev_pos[0]
        else:
            diff = None #no ball found
        ball_dir=diff
    return ball_dir, obs   

def evaluate_policy(env, policy, n_episodes=20,progress=1):
    total_rewards = 0.0
    if progress < 1/2:
        max_episodes = 1000 #reduced len to nudge learning in right direction in the beginning
    else:
        max_episodes = 100000 #just a big num which never will limit the episode
    for _ in range(n_episodes):
        total_rewards += run_episode(env, policy,render=False,episode_len=max_episodes)
    avg_score = total_rewards / n_episodes
    print("avg score: ", avg_score)
    return  avg_score
  
def process_image(full_image):
    processed_image = full_image[34:194] #crop image
    processed_image = np.delete(processed_image,list(range(0,16))+list(range(144,170)), axis=1) #remove sides
    processed_image = processed_image[:,:,0]   # remove the second and third RGB value
    mask = binary_dilation(processed_image==236,[np.ones(255)])
    mask[:,[0,1,2,3]]=0 # don't overwrite opponents line
    mask[:,[-1,-2,-3,-4]]=0 # don't overwrite own line 
    processed_image[mask]=236 #increase length of the ball
    processed_image = downsample_image(processed_image)
    processed_image = np.delete(processed_image,list(range(0,15))+list(range(16,31)), axis=1) #significantly 'simplify' the image
    processed_image[processed_image==144] = 0 # Set background to 0
    processed_image[processed_image==236] = -1 # Set ball to -1
    processed_image[processed_image==92] = 1  # Set cursor to 1
    return processed_image
   
def downsample_image(image):
    return image[::4,::4]
   
if __name__ == '__main__':
    n_policies = 12
    n_steps = 30
    env = gym.make('Pong-v0')
    env = gym.wrappers.Monitor(env, '/tmp/pong-v0', force=True, video_callable=lambda episode_id: episode_id%1000==0)
    random.seed(123456)  
    obs = env.reset()
    start = time.time()
    policy_set = [gen_random_policy() for _ in range(n_policies*2)]

    for idx in range(n_steps):
        policy_scores = [evaluate_policy(env, p, progress=(idx/n_steps)) for p in policy_set]
        print('Generation %d : max score = %0.2f' %(idx+1, max(policy_scores)))
        policy_ranks = list(reversed(np.argsort(policy_scores)))
        elite_set = [policy_set[x] for x in policy_ranks[:2]] #keep 2 best
        common_set = [policy_set[x] for x in policy_ranks[:(n_policies-4)]] # use all exept worst 4 for breeding
   
        for n in range(n_policies-4):
            pick1 = np.random.choice(n_policies-4)
            pick2 = np.random.choice(n_policies-4)
            if (pick1==pick2): # dont allow cloning...
                if (pick1==0):
                    pick1=1
                else:
                    pick1 = pick1-1           
            if (n==0):
                child_set = [crossover(common_set[pick1],common_set[pick2])]
            else:   
                child_set += [crossover(common_set[pick1],common_set[pick2])]
                
        mutated_list = [mutation(p) for p in child_set]
        elite_set_mutated = [mutation(p) for p in elite_set]
        policy_set = elite_set
        policy_set += elite_set_mutated
        policy_set += mutated_list

    policy_score = [evaluate_policy(env, p) for p in policy_set]
    best_policy = policy_set[np.argmax(policy_score)]
    print("best policy")
    print(best_policy)
    end = time.time()
    print("Best score = %0.2f. Time taken = %4.4f seconds" %(np.max(policy_score) , end - start))
    #end of learning

    # run best policy ~100 times to get best score average from gym
    for _ in range(120):
        run_episode(env, best_policy)
    env.close()