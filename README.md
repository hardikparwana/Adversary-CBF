# Multi-agent Connectivity and Control with CBF


This repository implements the work done during the term September-December 2022. 

## Description
This work considers a group of agents that are subjected to connectivity constraints and multiple safety constraints. The connectivity is either imposed by enforcing maximum distance constraint w.r.t leader agent OR by using gradient of the eigenvalue of laplacian matrix of tne connectivity graph. Both the approaches are implemented.

**Note:** While the theory in report has been goven for general dynamical systems, most implementations are still done for 2D single integrator agents only.

## Requirements
The code has been developed and tested in Python 3.8.16. A virtual environment can be first created using
```
python3.8 -m virtualenv venv
```
To activate the virtual environment
```
source venv/bin/avctivate
```
To deactivate, just run `deactiave`. To install dependencies, run
```
pip install -r requirements.txt
```
**Note:** The code uses `cvxpy` to frame and solve the Quadratic Program based controller. It uses GUROBI under the hood but if Gurobi is not available, simply remove the phrase `solver=cp.GUROBI` from all the codes. it will then use the free solver that comes with cvxpy.

## How to run the code

The examples files are present in the root folder. The robots and obstacles are defined as classes and their source code is included in the folder `robot_models`. Therefore, to make, for example, a single integrator model robot and obstacle, we simply import the following
```
from robot_models.SingleIntegrator2D import *
from robot_models.obstacles import *
```
The `obstacles.py` has rectangle and circles implemented. Only circles are used for simulation resul though.

## Simulation Results

### Connectivity Using Eigenvalue of Laplacian Matrix

- 1 leader: CBF for collision avoidance and eigenvalue gradient as reference input
```
python corridor.py
```

- 2 diverging leaders: CBF for collision avoidance + CBF for minimum eigenvalue + eigenvalue gradient as reference input
```
python corridor_2leaders.py
```

### Connectivity by maximum distance to leader constraint
Each agent uses CBFs for enfocing collision avoidance with other agents and one more CBF for enforcing maximum allowable distance to the leader

- 1 leader: pass through obstacle
```
python obstacle_leader.py
```
