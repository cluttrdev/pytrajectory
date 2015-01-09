# test example: double integrator

from pytrajectory import Trajectory
from pytrajectory.log import Timer

from IPython import embed as IPS
import numpy as np

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

T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, ua=ua, ub=ub, constraints=constraints)

T.setParam('kx', 3)
T.setParam('maxIt', 5)
T.setParam('eps', 1e-2)
T.setParam('ierr', 1e-2)
T.setParam('use_chains', False)
#T.setParam('sx', 100)


with Timer("Iteration"):
    T.startIteration()


IPS()
