from unityagents import UnityEnvironment
import numpy as np
import time

from DQN_tf import DeepQNetwork as DeepQNetwork_tf
from DQN_torch import DeepQNetwork as DeepQNetwork_torch

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # or any '0,1'


def train_agent(agent,
                env,
                n_episodes=2000,
                eps_start=1.0,
                eps_end=0.01,
                eps_decay=0.995):
    # get the default brain
    brain_name = env.brain_names[0]

    # try:
    #     agent.restore()
    #     print("model restored from global step {}".format(agent.learn_step_counter))
    # except ValueError:
    #     print("new training")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_file_name = "./models/DQN_torch_train_log-" + timestr + ".csv"

    eps = eps_start
    for i_episode in range(n_episodes):

        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        while True:
            action = agent.act(state, eps=eps)

            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

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

        if i_episode % 50 == 49:
            agent.save()


def test_agent(agent, env):
    brain_name = env.brain_names[0]

    try:
        agent.restore()
        print("model restored from global step {}".format(agent.learn_step_counter))
    except ValueError:
        print("failed to load")

    score_logger = []
    for i_episode in range(100):

        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        while True:
            action = agent.act(state, eps=0.01)

            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished

            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                print("Episode {}, Score: {}".format(i_episode, score))
                break

        score_logger.append(score)

    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(score_logger)), score_logger)
    plt.ylabel('score')
    plt.xlabel('episode')
    plt.show()


if __name__ == "__main__":
    # env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64", no_graphics=True)
    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # ---------------------------
    # env info
    # ---------------------------
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    # agent = DeepQNetwork_tf(n_features=state_size,
    #                         n_actions=action_size,
    #                         memory_size=int(1e5),
    #                         batch_size=64,
    #                         gamma=0.99,
    #                         learning_rate=0.0005,
    #                         update_every=100)

    agent = DeepQNetwork_torch(state_size=state_size,
                               action_size=action_size,
                               memory_size=int(1e5),
                               batch_size=64,
                               gamma=0.99,
                               learning_rate=0.0005,
                               update_every=100)

    # train_agent(agent, env)
    test_agent(agent, env)
    env.close()
