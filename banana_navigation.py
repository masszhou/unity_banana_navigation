from unityagents import UnityEnvironment
import numpy as np
import time

from DQN_tf import DeepQNetwork

from DQN_torch import DeepQNetwork as DeepQNetwork_torch

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # or any '0,1'


env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64", no_graphics=True)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# ---------------------------
# env info
# ---------------------------
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

# agent = DeepQNetwork(n_actions=action_size,
#                      n_features=state_size,
#                      learning_rate=0.01)

agent = DeepQNetwork_torch(state_size=state_size, action_size=action_size, seed=42)

N_EPISODES = 50000
scores_log = []

# try:
#     agent.restore()
# except IOError:
#     print("new training")


timestr = time.strftime("%Y%m%d-%H%M%S")
log_file_name = "./models/DQN_train_log-" + timestr + ".csv"

eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995
eps = eps_start
for i_episode in range(N_EPISODES):

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    while True:
        #action = agent.choose_action(state)  # select an action
        action = agent.act(state, eps=eps)

        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished

        #agent.store_transition(state, action, reward, next_state)
        #agent.learn()
        agent.step(state, action, reward, next_state, done)

        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        if done:  # exit loop if episode finished
            print("Episode {}, Score: {}, eps: {}".format(i_episode, score, eps))
            break
    with open(log_file_name, 'a') as file:
        info = "Episode {}, Score: {}, eps: {}\n".format(i_episode, score, eps)
        file.write(info)

    # if i_episode % 10 == 0:
    #     agent.save()

env.close()