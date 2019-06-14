import torch as t
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from collections import deque
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()  # equivalent to nn.Module.__init__(self)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class DQN:
    def __init__(self,
                 state_size,
                 action_size,
                 gamma=0.99,
                 learning_rate=0.0005,
                 update_every=4,
                 batch_size=64,
                 memory_size=int(1e5),
                 model_save_path="./models/",
                 log_save_path="./logs/",
                 output_graph=True):

        super(DQN, self).__init__()
        # ------------------------------------------
        # model parameters
        # ------------------------------------------
        self.action_size = action_size
        self.state_size = state_size
        self.lr = learning_rate
        self.gamma = gamma  # reward discounter
        self.update_every = update_every
        self.memory_size = memory_size
        self.batch_size = batch_size
        # initialize zero memory [s, a, r, s_, done]
        self.memory = deque(maxlen=self.memory_size)
        # total learning step
        self.learn_step_counter = 0

        # ------------------------------------------
        # model definition
        # ------------------------------------------
        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.optimizer = t.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def act(self, state, eps):
        if np.random.uniform() < eps:
            action = np.random.randint(0, self.action_size)
        else:
            # forward feed the state and get q value for every actions
            with t.no_grad():
                actions_value = self.policy_net(t.from_numpy(state).unsqueeze(0))  # batch size = 1
            action = np.argmax(actions_value.numpy())
        return action

    def step(self, s, a, r, next_s, done):
        # assert s,s_next -> np.array
        # assert a -> scalar
        # assert r -> scalar
        # assert done -> scalar
        self.memory.append(np.concatenate([s, [a], [r], next_s, [done]]))  # deque
        memory_size = len(self.memory)
        if memory_size < self.batch_size:
            return

        if self.learn_step_counter % self.update_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        sample_index = np.random.choice(range(memory_size), size=self.batch_size)  # replace = False
        batch_memory = np.array([self.memory[i] for i in sample_index])

        s_sample = batch_memory[:, 0:self.state_size]                              # shape=(#batch, n_features)
        a_sample = batch_memory[:, self.state_size:self.state_size+1].squeeze()    # shape=(#batch,)
        r_sample = batch_memory[:, self.state_size+1:self.state_size+2].squeeze()  # shape=(#batch,)
        s_next_sample = batch_memory[:, self.state_size+2:self.state_size*2+2]     # shape=(#batch, n_features)
        done_sample = batch_memory[:, self.state_size*2+2:self.state_size*2+3]     # shape=(#batch,)

        # Q[s][a] = Q[s][a] + alpha * (r + gamma * np.max(Q[s_next]) - Q[s][a])
        # Q_target = r + gamma * np.max(target_net(next_s))
        # Q_policy = policy_net(s)[a]
        # find fix point of Q*=BQ*, thus min norm2(Q_target - Q_policy)

        Q_target = r_sample + self.gamma * t.max(self.target_net(t.from_numpy(s_sample))) * (1 - done_sample)
        # no future reward when game done

        a_index = t.from_numpy(a_sample.astype(np.int64)).view([-1, 1])
        Q_policy = self.policy_net(t.from_numpy(s_next_sample)).gather(1, a_index)

        loss = F.mse_loss(Q_target, Q_policy)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
