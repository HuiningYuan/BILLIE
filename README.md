# BILLIE
This is a primitive sample code for the paper "Bi-level identification of complex dynamical systems through reinforcement 
learning" for initial submitting.
A more mature version of BILLIE will be publicly available on github after the paper is accepted.

This demo code contains two instances of identifying the burgers' equation and the Navier-Stokes equation with our proposed 
approach, named BILLIE (Bi-level identification of equations). 

Requirements
This Python code was tested under the following environment:
Python==3.8.5
Pytorch==1.7.1
numpy==1.20.3
scipy==1.6.2

Instance1: identifying the Burgers' equation

    Two sets of simulated data of the equation "u_t = 0.1*u_xx - u*u_x" with 5% Gaussian noise is included in the data folder, 
    corresponding to dataset s1 and s2 from the paper.
    
    To run this case on cpu, run: python main.py --data Burgers
    
    To run this case on gpu, run: python main.py --data Burgers --gpu 0

Instance2: identifying the 2D Navier-Stokes equation with a Reynolds number of 1000

    Since the original data for this case is too big (2048 x 2048 x 40 x 3 float64 numbers), the U_t and library Gamma of 30000 
    measurements was pre-built and included in the data folder.
    
    The two ground truth equations: u_t = 0.001*u_xx + 0.001*u_yy - p_x - 0.1*u - u*u_x - v*u_y
                                                         v_t = 0.001*v_xx + 0.001*v_yy - p_y - 0.1*v - v*v_x - u*v_y
                                                         
    To run this case on cpu, run: python main.py --data Navier-Stokes
    
    To run this case on gpu, run: python main.py --data Navier-Stokes --gpu 0

A more mature version of BILLIE will be available when the paper is published.
