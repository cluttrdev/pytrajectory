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
from log import logging

__version__ = '0.4'
__release__ = '0.4.0'

# Placeholder for the datetime string of latest commit
__date__ = "2015-02-17 18:00:49"

##################################
# check versions of dependencies #
##################################
import numpy
import scipy
import sympy

dependencies = [numpy, scipy, sympy]

np_info = numpy.__version__.split('.')
scp_info = scipy.__version__.split('.')
sp_info = sympy.__version__.split('.')

if not (int(np_info[0]) >= 1 and int(np_info[1]) >= 8):
    logging.warning('numpy version ({}) may be out of date'.format(numpy.__version__))
if not (int(scp_info[0]) >= 0 and int(scp_info[1]) >= 13 and int(scp_info[2][0]) >= 0):
    logging.warning('scipy version ({}) may be out of date'.format(scipy.__version__))
if not (int(sp_info[0]) >= 0 and int(sp_info[1]) >= 7 and int(sp_info[2][0]) >= 5):
    logging.warning('sympy version ({}) may be out of date'.format(sympy.__version__))

# is the following really necessary?
del numpy, scipy, sympy

