# ADCHEM ML and MPC Workshop 2021

This page contains the code associated with the ADCHEM 2021 Workshop on Machine Learning (ML) and Model Predictive Control (MPC). More information about the workshop can be found [here](https://www.adchem2021.org/workshop-machine-learning).

The code is written in Matlab and is broken down into 3 parts: 
1. Learning plant-model mismatch using Gaussian processes;
2. Learning deep neural network approximations of nonlinear MPC laws; 
3. Learning the optimal nonlinear MPC tuning parameters using constrained Bayesian optimization. 

The following packages are required
* [CasADi](https://web.casadi.org) must be installed, as we use its automatic differentiation and optimal control capabilities to easily formulate and solve the specified NMPC problems. 
* The [n4sid](https://www.mathworks.com/help/ident/ref/n4sid.html) function, which is a part of the Systems Identification Toolbox in Matlab, must be installed to perform the model identification step in Part 1.
* The [fitnet](https://www.mathworks.com/help/deeplearning/ref/fitnet.html) function, which is a part of the Deep Learning Toolbox in Matlab, must be installed to perform the deep neural network training in Part 2. 
* The [bayesopt](https://www.mathworks.com/help/stats/bayesopt.html) function, which is a part of the Statistics and Machine Learning Toolbox in Matlab, must be installed to execute the constrained Bayesian optimization (CBO) algorithm in Part 3. Note that there are alternative open-source implementations of CBO including [COBALT](https://github.com/joelpaulson/COBALT#readme). 
