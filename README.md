# Adversary-CBF


This repository implements our CDC 2022 submission on 

**Trust-based Rate-Tunable Control Barrier Functions for Non-Cooperative Multi-Agent Systems**

Authors: Hardik Parwana and Dimitra Panagou, University of Michigan

A pre-print version is available [here](https://arxiv.org/abs/2204.04555).

Note: this repo is under development. While all the relevant code is present, we will work on making it more readable and customizable soon! Stay Tuned!

## Description
This work considers the question: 

How to identify an unknown robot in the system as being:
1. Cooperative: i.e., it actively tries to maintain safety requirements
2. Aeversarial: i.e., it tries to chase and actively violate safety requirements
3. Uncooperative: i.e., it disregards any interaction with other agent and only moves toward's it's own goal. This is like saying: "Stay away from me and you won't be harmed!"

How can an autonomous agent adapt to these different types of robots? Treating all unknown agents in similar fashion can lead to conservative response. For example, a controller might try to make the agent back-off from an adversarial agent so as to avoid collision in future. However, if it were to know that the other agent is uncooperative, then it can adjust it's parameters to allow going closer to the other robot, slowing down enough to let the other robot pass first, and then continue on it's own path. This would cause minimal deviation from nominal trajectories.

This work adapts parameters of Control Barrier Functions and this page provides supplementary plots and videos to our submission.

# Simulation Results

| Trust based adaptation | Constant CBF parameter(small) | Constant CBF parameter(large) |
| --------------| -------------------| -----------------|
| ![PAPER_with_trust](https://user-images.githubusercontent.com/19849515/162593597-f028c61d-7a9d-4ff9-88b4-5851aeae1806.gif) | ![PAPER_NO_TRUST](https://user-images.githubusercontent.com/19849515/162593600-273fd93a-c82c-4655-b232-a03181672b15.gif) | ![PAPER_NO_TRUST_large_alpha](https://user-images.githubusercontent.com/19849515/162593605-af184d72-0d08-4c7e-bcdf-f88d18b42a5d.gif)


Above we show results for 3 cases
1. Trust based Adaptation: The CBF parameters are adapted based on trust mertic proposed in paper
2. Constant CBF parameter(small): The CBF parameters are kept same. Their values are same as the initial values of parameters in Trust-based adaptation
3. Constant CBF parameter(large): if you think having large CBF parameter that always allow close approach to other agents would work, then check this. The CBF parameter is about 3 times of that used before.

Simulation settings:
- 3 intact agents: these robots implement the proposed algorithm. Each tries to move vertically up while trying not to collide with any other agent. None of these robots know the identity of any other agnet (including the other 2 intact agent besides themselves). They use the proposed algorithm to update the CBF parameter corresponding to barrier constraint of every other agent individually. These follow unicycle dynamics and . Green circles show their location and lines originitaing from these circles show their heading direction (the heading angle psi).
- 2 uncooperative red robots: these move horizontally at constant speed without regard to movements of any other agent
- 1 adversarial red robot: this follows single integartor dynamics and chases one of the intact agent with a rate of 1 m/s.




# Dependencies
The code was run on Ubuntu 20 with Python 3.6 and following packages
- cvxpy==1.2.0
- gurobipy==9.5.1
- matplotlib==3.5.1
- numpy==1.22.3
- osqp==0.6.2.post5
- fonttools==4.31.2

In addition to above dependencies, run `source export_setup.sh` from main folder to set the paths required to access submodules.

# Running the Simulation
To simulate the scenario in paper run
```
python paper_task.py
```

