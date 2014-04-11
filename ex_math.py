# Example from OPTINUM : Beleg 3

from pytrajectory import Trajectory
import numpy as np
from sympy import  sin, cos

def f(x):
    x1, x2, x3 = x
    
    ff = np.array([             x2,
                    sin(x3)*x1 - 4*x2 + cos(x3),
                                1.0
                    ])
    
    return ff

xa = [  -0.12,
        -5.0,
         0.0]

xb = [  -0.12,
        -5.0,
        np.pi]

a, b = (0.0, np.pi)

T = Trajectory(f, a, b, xa, xb)

T.startIteration()
