from unityagents import UnityEnvironment
import numpy as np
import time

from DQN_tf import DeepQNetwork

env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

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

agent = DeepQNetwork(n_actions=action_size,
                     n_features=state_size,
                     learning_rate=0.01)

N_EPISODES = 50000
scores_log = []

try:
    agent.restore()
except:
    print("new training")


timestr = time.strftime("%Y%m%d-%H%M%S")
log_file_name = "./models/DQN_train_log-" + timestr + ".csv"

for i_episode in range(N_EPISODES):

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    while True:
        action = agent.choose_action(state)  # select an action

        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished

        agent.store_transition(state, action, reward, next_state)
        agent.learn()

        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            print("Episode {}, Score: {}, global steps: {}".format(i_episode, score, agent.learn_step_counter))
            break
    with open(log_file_name, 'a') as file:
        info = "Episode {}, Score: {}, global steps: {}".format(i_episode, score, agent.learn_step_counter)
        file.write(info)

    if i_episode % 10 == 0:
        agent.save()

env.close()