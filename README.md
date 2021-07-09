# CBF-Adversary

## Dependencies
This code is written in MATLAB. Follwoing additional packages need to be installed
1. [gurobi](https://www.gurobi.com/documentation/9.1/quickstart_mac/software_installation_guid.html): optimization library used for solving QPs
2. [cvx](http://cvxr.com/cvx/): library for disciplined optimization. Makes life easier by providing interface to write optimization problem. No need to form matrices on our own. **cvx** can use many optimizatiuon solvers under the hood. This project uses gurobi which can be set as default solver by typing followinbg two commands on MATLAB command line ([http://cvxr.com/cvx/doc/gurobi.html](http://cvxr.com/cvx/doc/gurobi.html)):
```
cvx_solver gurobi
cvx_save_prefs
```

## Code organization
1. **predefined_functions**: contains helper functions
2. **predefined classes**: this project defines robots and obstacles as class objects so that the main code can easily be modified for different robots and environments:
      * Unicycle2D: class for implementing Unicycle dynamics. see **multi_robot.m** for example usage.
      * EnvObject2D.m: class for making circular and rectangular objects in environment
      
       

More robots and objects will be added in future.

## Run
Simply add subfolders to MATLAB path and then run **multi_robot.m**
