- [Project: Banana Navigation](#Project-Banana-Navigation)
- [1. Introduction](#1-Introduction)
- [2. Environment](#2-Environment)
- [3. Basic Reinforcement Learning Model](#3-Basic-Reinforcement-Learning-Model)
  - [3.1 State Value Function](#31-State-Value-Function)
  - [3.2 Action Value Function](#32-Action-Value-Function)
  - [3.3 Bellman Equation](#33-Bellman-Equation)
- [4. Convergence for Value-Based Reinforcement Learning](#4-Convergence-for-Value-Based-Reinforcement-Learning)
  - [4.1 Bellman Operator](#41-Bellman-Operator)
  - [4.2 Contraction Mappings](#42-Contraction-Mappings)
  - [4.3 Reference for this section](#43-Reference-for-this-section)
- [5. Q-Learning](#5-Q-Learning)
  - [5.1 Temporal Difference Learning](#51-Temporal-Difference-Learning)
- [6. Deep Q Network](#6-Deep-Q-Network)
    - [6.1 Problem definition](#61-Problem-definition)
  - [6.2 Layers](#62-Layers)
- [7. Training Deep Q Network](#7-Training-Deep-Q-Network)
- [8. future works](#8-future-works)
- [References](#References)

# Project: Banana Navigation

# 1. Introduction

Train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent](https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif)]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

* The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  
    * state space has 37 dims. [0-34] for perception, [35,36] for velocity
    * the agent has 7 perception ray, each is a 5-dims vector
    * 7 rays projecting from the agent at the following angles (and returned in this order): [20, 90, 160, 45, 135, 70, 110]
    * where 90 is directly in front of the agent
    * Each ray is projected into the scene. If it encounters one of four detectable objects the value at that position in the array is set to 1. 
    * Finally there is a distance measure which is a fraction of the ray length. something like this [Banana, Wall, BadBanana, Agent, Distance]
    * For example [0, 1, 1, 0, 0.2] means there is a BadBanana detected 20% of the way along the ray and a wall behind it.
    * Velocity of Agent (2)
        * Left/right velocity (usually near 0)
        * Forward/backward velocity (0-11.2)
    * [discussion](https://knowledge.udacity.com/questions/22697) on udacity
    * [discussion](https://github.com/Unity-Technologies/ml-agents/issues/1134) on unity banana environment
* Four discrete actions are available, corresponding to:
    - **`0`** - move forward.
    - **`1`** - move backward.
    - **`2`** - turn left.
    - **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

# 2. Environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the unzip folder in the root of this project. 


# 3. Basic Reinforcement Learning Model
> The mathmatic behind reinforcement learning is HARD!

<img src="./imgs/RL_model.png"  width="600" />

* Reinforcement Learning is about learning **Policy** from the interaction between agent and environment
* A **Policy** function, written like $p=\pi(s,a)$, describes how the agent act. It reads as the probability of agent to take action $a$ at state $s$.
* The interaction of agent and environment can be described as a sequence of 
  * $S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}, R_{t+2}, S_{t+2}, \dots $
  * It read as the agent at **State** $S_t$ made an **Action** $A_t$, the environments gave a instance feedback **Reward** $R_{t+1}$ and subsequent **State** $S_{t+1}$ based on **State** $S_t$ and the **Action** $A_t$.
* Most of the time, we are more interested for maximizing cumulative future rewards, such as win a game at last. So we denote cumulative rewards at time step $t$ as 
<a href="https://www.codecogs.com/eqnedit.php?latex=G_t=R_{t&plus;1}&plus;\gamma&space;R_{t&plus;2}&space;&plus;&space;\gamma^2&space;R_{t&plus;3}&space;\dots&space;=&space;\sum^{\infty}_{k=0}\gamma^k&space;R_{t&plus;k&plus;1},&space;r\in[0,&space;1]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?G_t=R_{t&plus;1}&plus;\gamma&space;R_{t&plus;2}&space;&plus;&space;\gamma^2&space;R_{t&plus;3}&space;\dots&space;=&space;\sum^{\infty}_{k=0}\gamma^k&space;R_{t&plus;k&plus;1},&space;r\in[0,&space;1]" title="G_t=R_{t+1}+\gamma R_{t+2} + \gamma^2 R_{t+3} \dots = \sum^{\infty}_{k=0}\gamma^k R_{t+k+1}, r\in[0, 1]" /></a>

  * $G_t$ is also called **Return** at time step $t$
  * $\gamma$ is **Discount Rate**


## 3.1 State Value Function
* definition
$$
v_{\pi}(s)=\mathop{\mathbb{E}}[G_t|S_t=s]
$$
* read: the expected cumulative return from state $s$ following policy $\pi$
* consider all following rewards are known in a finite episode
* can be used to evaluate policy, e.g.
$$
\pi'>\pi \iff v_{\pi'}(s)\geq v_{\pi}(s), \forall s\in \mathcal{S}
$$
  * $\pi'$ is better than $\pi$, when $v_{\pi'}(s)\geq v_{\pi}(s), \forall s\in \mathcal{S}$

## 3.2 Action Value Function
* definition
$$
q_{\pi}(s,a)=\mathop{\mathbb{E}}[G_t|S_t=s, A_t=a]
$$
* read: the cumulative return from state $s$ when take action $a$ and subsequently following policy $\pi$
* a bridge between state value, and action, policy

## 3.3 Bellman Equation
* Redefine state value function and action value function in an iterative way

* Bellman expectation equation
$$
v_{\pi}(s)=\mathop{\mathbb{E}}[R_{t+1}+\gamma v_{\pi}(s_{t+1})|S_t=s] 
$$
  * or use general symbol $s, s'$ stands for two states. expend Bellman expectation equation, then we have
$$
v_{\pi}(s)=R(s,\pi(s))+\gamma \sum_{s'\in \mathcal{S}} p(s'|s,\pi(s)) v_{\pi}(s')
$$
  * or denote $\mathcal{P}^{\pi}_{ss'}=p(S_{t+1}=s' | S_t=s, A_t=\pi(s))$, which is more compact for matrix form
  $$
  v_{\pi}(s)=R^{\pi}_s+\gamma \sum_{s,s'\in \mathcal{S}}  \mathcal{P}^{\pi}_{ss'} v_{\pi}(s')
  $$
* Bellman optimal equation
$$
v_*(s)=\max_{a\in\mathcal{A}}\mathop \lbrace R^a_s + \gamma \sum_{s,s'\in \mathcal{S}}\mathcal{P}^a_{ss'}v_*(s') \rbrace
$$

# 4. Convergence for Value-Based Reinforcement Learning
## 4.1 Bellman Operator
* recall Bellman expectation equation
$$
v_{\pi}(s)=R^\pi_{s}+\gamma \sum_{s,s'\in \mathcal{S}}  \mathcal{P}^{\pi}_{ss'} v_{\pi}(s')
$$
* for all states, we can rewrite above equation in matrix form like
$$
\begin{bmatrix} v(1) \\ \vdots \\ v(n) \end{bmatrix} = \begin{bmatrix} R_1 \\ \vdots \\ R_n \end{bmatrix} + \gamma \begin{bmatrix} \mathcal{P}_{11} & \dots & \mathcal{P}_{1n} \\ \vdots \\ \mathcal{P}_{n1} & \dots & \mathcal{P}_{nn} \end{bmatrix} \begin{bmatrix} v(1) \\ \vdots \\ v(n) \end{bmatrix}
$$
* or more compactly
$$
v_{\pi}=\mathcal{R}^{\pi}+\gamma \mathcal{P}^{\pi}v_{\pi}
$$
* define Bellman expectation operator $\mathcal{T}^{\pi}:\mathbb{R}^n \rightarrow \mathbb{R}^n$ as 
$$
\mathcal{T}^{\pi}v_{\pi} = \mathcal{R}^{\pi}+\gamma \mathcal{P}^{\pi}v_{\pi}
$$

* similarly we have Bellman optimality operator
$$
\mathcal{T}^*v = \max_{a\in\mathcal{A}}(\mathcal{R}^a+\gamma\mathcal{P}^av)
$$

## 4.2 Contraction Mappings
* Bellman operator is a contraction mapping. (a lot of read... if I ask why. and I asked)
* $v_{\pi}$ and $v_{\ast}$ are unique fixed points. By repeatedly applying $\mathcal{T}^{\pi}$ and $\mathcal{T}^*$ they will converge to respectively
$$
\begin{aligned} 
\lim_{k\rightarrow\infty}(\mathcal{T}^{\pi})^kv&=v_{\pi} \\
\lim_{k\rightarrow\infty}(\mathcal{T}^{\ast})^kv&=v_{\ast}
\end{aligned}
$$

## 4.3 Reference for this section
* [What is the Bellman operator in reinforcement learning?](https://ai.stackexchange.com/questions/11057/what-is-the-bellman-operator-in-reinforcement-learning)
* [How Does Value-Based Reinforcement Learning Find the Optimal Policy?](https://runzhe-yang.science/2017-10-04-contraction/)


# 5. Q-Learning
* also called maximum SARS
* Asynchronous value iteration
* off-policy learning, use max(Q_next) to learn instead of using orignial policy $\pi$
* update Q-table $q_{\pi}$ with $(s_t, a_t, r_{t+1}, s_{t+1})$,
$$
q_{\pi}(s_t, a_t) = q_{\pi}(s_t, a_t) + \alpha  (r_{t+1} + \gamma \max q_{\pi}(s_{t+1}) - q_{\pi}(s_t, a_t))
$$
* or can be rewritten like soft-update form
$$
q_{\pi}(s_t, a_t) = (1-\alpha)q_{\pi}(s_t, a_t) + \alpha (r_{t+1} + \gamma \max q_{\pi}(s_{t+1}))
$$

## 5.1 Temporal Difference Learning
* learning in time, not until episode end.
* I will finished this summary later. I wish the summary intuitive but at the same time mathematical reasonable.


# 6. Deep Q Network
* use neural network to fit high dimensional Q-table

### 6.1 Problem definition
* recall Q-learning
$$
q_{\pi}(s_t, a_t) = (1-\alpha)q_{\pi}(s_t, a_t) + \alpha (r_{t+1} + \gamma \max q_{\pi}(s_{t+1}))
$$
* considering both contraction mappings and temporal difference.
  * let $q_{\pi}(s_t, a_t)$ be the temporal optimal state value
  * let $r_{t+1} + \gamma * \max q_{\pi}(s_{t+1})$ be the iterative Bellman operator part
* we know that by repeatly applying $\mathcal{T}^*$, $\mathcal{T}^*v$ will converge to $v_*$
* So we have optimization problem
$$
\min ||\mathcal{T}^*v - v_*||
$$
* in the view of temporal difference, we redefine the optimization problem within a small interval, like
$$
\min_{\text{policy}} \vert\vert\text{policynet} - \text{targetnet}\vert\vert
$$
* so we define two networks
  * one policy network (also called as evaluation network)
  * one target network
* we iterative update policy net to minimize $||\text{policynet} - \text{targetnet}||$ by training neural network.
* and periodically copy policy net to target net, due to temporal difference learning

## 6.2 Layers
* use dense layers to fit high dimensional Q-table
* in my experiments, I used
```
net = dense(state_size, 64)(x)
net = dense(64, 64)(net)
net = dense(64, action_size)(net)
```

# 7. Training Deep Q Network
* when hard copy from policy net to target net is too frequent, e.g. every 4 learning steps, the final performance will not be consistence due to my experiements. For example, one run will converge at early stage well, but some other runs will not converge.

* the agent will sometimes stuck, which can be imagined that there could be dead loop inside Q-table or network.

* the network capacity is important, e.g. layers and neurons. The network capacity can be interpreted as Q-table dimension. If the task is complex, then we need a deeper network to fit.

* the reward system or design is one of the most important part in RL.

* here is the evaluation runs from my model
  * 100 episodes
  * average score is 16
  * max score is 25
  * min score is 0
<img src="./imgs/test_run.png"  width="305" />

* replay the results
```
$ python banana_navigation.py
```

# 8. future works
* Double DQN is designed to solve Q-value explode.
* Prioritized Experience Replay can improve performance of sparse problem.

# References
* Udactiy Deep Reinforcement Learning Nanodegree
* Udacity Reinforcement Learning by Prof. Charles Isbell and Prof. Michael Littman
* Tutorials from MorvanZhou
* other blogs, articles, a lot