# test example: double integrator

from IPython import embed as IPS
import numpy as np

from pytrajectory.system import ControlSystem
from pytrajectory.log import Timer

def f(x,u):
    x1, x2 = x
    u1, = u

    ff = np.array([ x2,
                    u1])
    return ff

xa = [0.0, 0.0]
xb = [1.0, 0.0]

a = 0.0
b = 2.0
ua = [0.0]
ub = [0.0]
constraints = { 1:[-0.1, 0.65]}
#constraints = dict()

S = ControlSystem(f, a=a, b=b, xa=xa, xb=xb, ua=ua, ub=ub, constraints=constraints)

S.set_param('eps', 1e-2)
S.set_param('ierr', 1e-1)
S.set_param('use_chains', False)

with Timer("Iteration"):
    S.solve()

IPS()