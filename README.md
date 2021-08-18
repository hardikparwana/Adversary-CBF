# CBF-Adversary
This repo gives exmaples of multiagent control with cvx library and also gives a library (in form of class) for Multivariate Gaussian Process(Matrix Normal Distribution [MVG](https://en.wikipedia.org/wiki/Matrix_normal_distribution)) Estimation.

## Dependencies
This code is written in MATLAB. Following additional packages need to be installed for muli_robot.m
1. [gurobi](https://www.gurobi.com/documentation/9.1/quickstart_mac/software_installation_guid.html): optimization library used for solving QPs
2. [cvx](http://cvxr.com/cvx/): library for disciplined optimization. Makes life easier by providing interface to write optimization problems. No need to form matrices on our own. **cvx** can use many optimizatiuon solvers under the hood. This project uses gurobi which can be set as default solver by typing following two commands on MATLAB command line ([http://cvxr.com/cvx/doc/gurobi.html](http://cvxr.com/cvx/doc/gurobi.html)):
```
cvx_solver gurobi
cvx_save_prefs
```

## Code organization
1. **predefined_functions**: contains helper functions
2. **predefined classes**: this project defines robots and obstacles as class objects so that the main code can easily be modified for different robots and environments:
      * Unicycle2D: class for implementing Unicycle dynamics. see **multi_robot.m** for example usage.
      * EnvObject2D.m: class for making circular and rectangular objects in environment
      * MatrixVariateGaussianProcess.m: class for prediction and hyperpameter tuning of multivariate Gaussian Process where the quantity to be predicted is a vector instead of scalar. See **test_MVG.m** for example usage.
      
       

More robots and objects will be added in future.

## Run
1. Simply add subfolders to MATLAB path and then run **multi_robot.m**
2. To test MVG, add subfolders to MATLAB path and then simply run **test_MVG.m**
