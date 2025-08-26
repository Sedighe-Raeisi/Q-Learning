# Cognetive Reinforcement Learning   
In this chapter, we see the implementation of this method for cognitive reinforcement Learning.
## Getting Started
If you face a problem with the graphiz path, you can use this instruction:   
[link](https://www.pythonclear.com/errors/failed-to-execute-windowspathdot-make-sure-the-graphviz-executables-are-on-your-systems-path/?form=MG0AV3)   
and change os command in plot_model.py
<!--In terminal run:
- pip install -q git+https://github.com/dynamicslab/pysindy.git
- pip install -q pyro-ppl
- pip uninstall numpyro
- pip install -q numpyro@git+https://github.com/pyro-ppl/numpyro
- pip install -U jax  
- pip install arviz
- conda install python-graphviz-->

## Problem 
In the RL approach to explain the data of the decision-making task, we use the choice, the reward, and a latent variable, q-value to explain the mechanism of making a decision.   

![image](https://github.com/user-attachments/assets/5e666ac2-bebf-4d5d-afb7-9c5e73f0edd8)
    

The deck with a greater value of Q-value is more probable to be selected.   
The Q-value at a trial or experiment is related to its previous values. Different mechanisms could affect the new Q-value. Some of them are:

![image](https://github.com/user-attachments/assets/d0ae6f7e-b8ff-4770-bebb-15be117bcfd7)


These mechanisms are different between different participants.   

![image](https://github.com/user-attachments/assets/b7b8ae0a-4e14-47aa-abcc-502003862c35)  

Some of them are:
- Shared, exist, and have the same coefficient or magnitude
- Shared, exist, with different coefficient or magnitude
- Shared, doesn't exist for all participants
- Non-shared, means that it exists for some individuals and doesn't exist for others.  

  ![image](https://github.com/user-attachments/assets/8ad247a7-1c75-4e87-a0db-1cdf766a56a7)





## 1- Uniform distribution, modeled by Non-Hierarchical Bayesian inference 
In the first step, we considered that the coefficient of each term is the same for all individuals in the population. We used the Bayesian-Sindy to extract the model that relates the Q-value to its previous value. 
In this approach, we prepare a list of all candidate terms for the dataset of y=Q and x=Q_1. These may include different possible linear combinations of x up to 2 orders. 
Then, by using the Horseshoe distribution as a prior for the coefficient of each term, we retrieve their posterior distribution.   

![image](https://github.com/user-attachments/assets/6f6ccd12-709b-45f0-9937-ee2e1c6fab8c)


It is shown that for non-existing terms, the posterior distribution will have a mean close to zero. 
For existing terms, the posterior distribution will have a mean approximately equal to its ground truth value.    

![image](https://github.com/user-attachments/assets/67faac76-54e2-46b8-b4c9-4ae5e11ce181)


## 2- Mixed distribution, modeled by Hierarchical Bayesian inference 
In the second step, we considered that the coefficient of some terms is different for each individual in the population. At first, we used non-hierarchical Bayesian SINDy. We observed that although the means of distribution of each term were successfully retrieved, the standard deviation wasn't recovered. Therefore, we used the Bayesian-Sindy with a Hierarchical structure to extract the model that relates the Q-value to its previous value.   

![image](https://github.com/Sedighe-Raeisi/BH_PKG/blob/main/Example/test_HS_model.png?raw=true)


  

  

We could see that both mean and standard deviation matched their corresponding ground truth values.   

![image](https://github.com/user-attachments/assets/68cc91a0-2960-4a37-a6ba-f333b067d608)


## 3- Non-shared mechanism 
In the third step, we consider a feature like forget-rate to be non-shared in the population. 
It means that for a portion of the population, the coefficient of this term is zero, for the rest of the population, we might consider different scenarios.  
It will result in the existence of a peak in the distribution alongside another distribution. 
The result for all features:  

![image](https://github.com/user-attachments/assets/78cb384c-1982-4a83-8a78-b8864916b997)
    

In this condition, we expect the model to predict the peak portion accurately. So the model should be able to recover :
- portion of the zero peak
- mean of non-zero distribution
- standard deviation of non-zero distribution

![image](https://github.com/user-attachments/assets/c6c828e8-24cc-45a0-8ee4-7339231eb4fc)


  

