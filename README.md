- [Project: Banana Navigation](#Project-Banana-Navigation)
- [1. Introduction](#1-Introduction)
- [2. Environment](#2-Environment)
- [3. Reinforcement Learning Model](#3-Reinforcement-Learning-Model)

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


# 3. Reinforcement Learning Model

* see [report.pdf](./report.pdf)

# 4. ToDO
* try Rainbow
