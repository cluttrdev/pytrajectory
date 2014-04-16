# translation of the inverse pendulum

# import trajectory class and necessary dependencies
from pytrajectory import Trajectory
from sympy import sin, cos
import numpy as np

# define the function that returns the vectorfield
def f(x,u):
    x1, x2, x3, x4 = x       # system state variables
    u1, = u                  # input variable

    l = 0.5     # length of the pendulum rod
    g = 9.81    # gravitational acceleration
    M = 1.0     # mass of the cart
    m = 0.1     # mass of the pendulum

    s = sin(x3)
    c = cos(x3)

    ff = np.array([                     x2,
                   m*s*(-l*x4**2+g*c)/(M+m*s**2)+1/(M+m*s**2)*u1,
                                        x4,
            s*(-m*l*x4**2*c+g*(M+m))/(M*l+m*l*s**2)+c/(M*l+l*m*s**2)*u1
                ])
    return ff

# boundary values at the start (a = 0.0 [s])
xa = [  0.0,
        0.0,
        0.0,
        0.0]

# boundary values at the end (b = 1.0 [s])
xb = [  0.5,
        0.0,
        0.0,
        0.0]

# create trajectory object
T = Trajectory(f, a=0.0, b=1.0, xa=xa, xb=xb)

# run iteration
T.startIteration()

# show results
T.plot()
