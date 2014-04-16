# underactuated manipulator

# import trajectory class and necessary dependencies
from pytrajectory.trajectory import Trajectory
import numpy as np
from sympy import cos, sin
from numpy import pi

# define the function that returns the vectorfield
def f(x,u):
    x1, x2, x3, x4  = x     # state variables
    u1, = u                 # input variable
    
    e = 0.9     # inertia coupling
    
    s = sin(x3)
    c = cos(x3)
    
    ff = np.array([         x2,
                            u1,
                            x4,
                    -e*x2**2*s-(1+e*c)*u1
                    ])
    
    return ff

# system state boundary values for a = 0.0 [s] and b = 1.8 [s]
xa = [  0.0,
        0.0,
        0.4*pi,
        0.0]

xb = [  0.2*pi,
        0.0,
        0.2*pi,
        0.0]

# boundary values for the inputs
g = [0.0, 0.0]

# create trajectory object
T = Trajectory(f, a=0.0, b=1.8, xa=xa, xb=xb, g=g)

# also alter some method parameters to increase performance
T.setParam('su', 20)
T.setParam('kx', 3)

# run iteration
T.startIteration()

# show results
T.plot()
