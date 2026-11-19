# COMS-4726-Neural-Bandit-Project
This repo is for the purpose of the final project of Columbia University FA 26 COMS 4726 Deep Learning for Data Science.

## Structure of the repo

### Reproduce
The directory contains scripts and interim data to reproduce the paper [**A Neural Networks Committee for the Contextual Bandit Problem** ](https://arxiv.org/abs/1409.8191). 

To run a smoke test, run `cd reproduce` then `python run.py --T 50000 --seeds 2 --out results_smoke` in the terminal. To fully run the experiment, run `python run.py` (note this may take certain amount of time to finish). However, it is not necessary to run the entire experiment to obtain the result plot as the interim results have been stored in reproduce/results/. 

In order to gain the result plot, run `python plot.py`. 


### Improvements
This directory contains scripts to improve the existing bandit committee structure. 
