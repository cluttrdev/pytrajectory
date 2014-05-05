from pytrajectory.trajectory import Trajectory
import pytrajectory.log as log

from IPython import embed as IPS

from sympy import cos, sin
from numpy import pi


# partiell linearisiertes inverses Pendel [6.1.3]

calc = False

def f(x,u):
    x1, x2, x3, x4 = x  # system variables
    u1, = u             # input variable
    
    l = 0.5     # length of the pendulum
    g = 9.81    # gravitational acceleration
    
    # this is the vectorfield
    ff = [          x2,
                    u1,
                    x4,
        (1/l)*(g*sin(x3)+u1*cos(x3))]
    
    return ff

a = 0.0
xa = [0.0, 0.0, pi, 0.0]

b = 2.0
xb = [0.0, 0.0, 0.0, 0.0]

uab = [0.0, 0.0]

T = Trajectory(f, a, b, xa, xb, uab)

T.setParam('kx', 5)
#T.setParam('eps', 0.05)
T.setParam('use_chains', False)

with log.Timer("startIteration"):
    T.startIteration()
