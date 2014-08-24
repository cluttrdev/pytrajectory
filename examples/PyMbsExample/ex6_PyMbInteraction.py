#!/usr/bin/python2

######################################################################
# This example demonstrates how to get motion equations of a 
# multibody system created with PyMbs and use them to determine a 
# feed forward control with PyTrajectoy


# PyMbs imports
from PyMbs.Input import *
from PyMbs.utils.sympyexport import sympymbs

# PyTrajectory imports
from pytrajectory import Trajectory

# This is now part of PyMbs
#from pytrajectory.utilities import sympymbs

# additional imports
import numpy as np

# The following import requires some explanation:
# 
# To add a controller to the multibody system (in this example a force
# that acts on the crab) the user has to provide a python module
# (here: dummy_controller) that contains a function (here: controlForce)
# that returns the state of the desired controller depending on the
# current time.
# 
# Since we want to determine this function it has to be just a dummy 
# procedure by now, that will be replaced with the solution found 
# (hopefully) by PyTrajectory.
# 
# For more information have a look at the file: dummy_controller.py
import dummy_controller
from dummy_controller import controlForce


######################################################################
# PyMbs
######################################################################

# Here we build a multibody system that is basically a simple pendulum.

# Create MbsSystem
world=MbsSystem([0,0,-1])

m1 = world.addParam('m1', 54)
m2 = world.addParam('m2', 2.65 )
m3 = world.addParam('m3', 38 )

# Add bodies
crab = world.addBody(mass=m1, cg=[0.24,0.02,0.21], inertia=diag([2.11,5.87,5.61]))
hook = world.addBody(mass=m2, cg=[0,0,0.08], inertia=diag([0.01,0.01,0.1]))
load = world.addBody(mass=m3, cg=[0,0,-1.28], inertia=diag([2.28,2.28,0.15]))

# Join bodies
world.addJoint(world, crab, 'Tx', startVals=1)
world.addJoint(crab, hook)
world.addJoint(hook, load, 'Ry')

# Add visualization of the system

filetyp = "stl_files/%s_01.stl"
world.addVisualisation.File(world, filetyp%'Traeger', name='Traeger')
world.addVisualisation.File(crab, filetyp%'Laufkatze', name='Laufkatze')
world.addVisualisation.File(hook, filetyp%'Haken', name='Haken' )
world.addVisualisation.File(load, filetyp%'Last', name='Last' )

# Add a control force
# -> by now this is just a dummy procedure that is necessary for PyMbs
#    to generate the motion equations (see: dummy_controller.py)
F = world.addController('F', controlForce, shape=(3, ))
world.addLoad.CmpForce(F, crab, world, name='DrivingForce')

# Determine motion equations
world.genEquations.Recursive()


######################################################################
# PyTrajectory 
######################################################################

# Get a function of the motion equations of the multibody system 
# that can be used with PyTrajectory
f = sympymbs(world)

# boundary values
a, b = 0.0, 2.0

xa = [0.0, 0.0, 0.0, 0.0]
xb = [0.0, np.pi, 0.0 ,0.0]

uab = [0.0, 0.0]

# create PyTrajectory object
T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, g=uab)

# change some method parameters
T.setParam('kx', 5)
T.setParam('use_chains', False)
T.setParam('eps', 0.05)
T.setParam('ierr', 5e-2)

# run iteration
xt, ut = T.startIteration()

# the following definition is necessary because the determined control
# function u(t) is just valid for t in (a,b)
def u_fnc(t):
    if a <= t <= b:
        return ut(t)
    else:
        return ut(b)

# now we replace the dummy procedure with the solution from PyTrajectory
dummy_controller.container[0]= u_fnc

# and visualise the system
world.show('Laufkatze')
