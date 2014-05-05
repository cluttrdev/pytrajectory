# oscillation of the inverted double pendulum with partial linearization

# import trajectory class and necessary dependencies
from pytrajectory.trajectory import Trajectory
from sympy import cos, sin
import numpy as np
from numpy import pi

# define the function that returns the vectorfield
def f(x,u):
	x1, x2, x3, x4, x5, x6 = x  # system variables
	u, = u                      # input variable
    
    # length of the pendulums
	l1 = 0.7
	l2 = 0.5
    
	g = 9.81    # gravitational acceleration
    
	ff = np.array([         x2,
                            u,
                            x4,
                (1/l1)*(g*sin(x3)+u*cos(x3)),
                            x6,
                (1/l2)*(g*sin(x5)+u*cos(x5))
                    ])
    
	return ff

# system state boundary values for a = 0.0 [s] and b = 2.0 [s]
xa = [0.0, 0.0,  pi, 0.0,  pi, 0.0]
xb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# boundary values for the input
uab= [0.0, 0.0]

# create trajectory object
T = Trajectory(f, a=0.0, b=2.0, xa=xa, xb=xb, g=uab)

# alter some method parameters to increase performance
T.setParam('su', 10)
T.setParam('eps', 8e-2)

# run iteration
T.startIteration()

# show results
T.plot()
