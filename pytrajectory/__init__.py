'''
PyTrajectory
============

PyTrajectory is a Python library for the determination of the feed forward control 
to achieve a transition between desired states of a nonlinear control system.
'''

from trajectory import Trajectory
from spline import CubicSpline
from solver import Solver
from simulation import Simulation
from utilities import Animation

__version__ = '0.3'
__release__ = '0.3.4'
