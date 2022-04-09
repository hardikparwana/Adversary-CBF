# Adversary-CBF


This repository implements our CDC 2022 submission on 

**Trust-based Rate-Tunable Control Barrier Functions for Non-Cooperative Multi-Agent Systems**

Authors: Hardik Parwana and Dimitra Panagou, University of Michigan

Note: this repo is under development. While all the relevant code is present, we will work on making it more readable and customizable soon! Stay Tuned!

## Description
This work considers the question: 

How to identify an unknown robot in the system as being:
1. Cooperative: i.e., it actively tries to maintain safety requirements
2. Aeversarial: i.e., it tries to chase and actively violate safety requirements
3. Uncooperative: i.e., it disregards any interaction with other agent and only moves toward's it's own goal. This is like saying: "Stay away from me and you won't be harmed!"

How can an autonomous agent adapt to these different types of robots? Treating all unknown agents in similar fashion can lead to conservative response. For example, a controller might try to make the agent back-off from an adversarial agent so as to avoid collision in future. However, if it were to know that the other agent is uncooperative, then it can adjust it's parameters to allow going closer to it, slow down enough to let the other robot pass first, and then continue on it's own path. This would cause minimal deviation from nominal trajectories.

This work adapts parameters of Control Barrier Functions and provides supplementary plots and videos to our submission.


# Dependencies
The code was run on Ubuntu 20 with Python 3.6 and following packages
- cvxpy==1.2.0
- gurobipy==9.5.1
- matplotlib==3.5.1
- numpy==1.22.3
- osqp==0.6.2.post5
- fonttools==4.31.2

In addition to above dependencies, run `source export_setup.sh` from main folder to set the paths required to access submodules.
