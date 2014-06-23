from PyMbs.Input import *
from IPython import embed as IPS
from dummy_controller import controlForce
import dummy_controller

from pytrajectory import Trajectory
from pytrajectory.utilities import sympymbs

import numpy as np

######################################################################
# This example demonstrates how to get motion equations of a 
# multibody system created with PyMbs and use them to determine a 
# trajectory with PyTrajectoy


######################################################################
# PyMbs
######################################################################

# MbsSystem anlegen
world=MbsSystem([0,0,-1])

m1 = world.addParam('m1', 54)
m2 = world.addParam('m2', 2.65 )
m3 = world.addParam('m3', 38 )
m4 = world.addParam('m4', 38 )

# Koerper definieren
crab = world.addBody(mass=m1, cg=[0.24,0.02,0.21], inertia=diag([2.11,5.87,5.61]))
hook = world.addBody(mass=m2, cg=[0,0,0.08], inertia=diag([0.01,0.01,0.1]))
load1 = world.addBody(mass=m3, cg=[0,0,-0.78], inertia=diag([0.0,0.0,0.0]))
load2 = world.addBody(mass=m4, cg=[0,0,-0.58], inertia=diag([0.0,0.0,0.0]))

# Koerper mit Gelenken verbinden
world.addJoint(world, crab, 'Tx', startVals=1)
world.addJoint(crab, hook)
world.addJoint(hook, load1, 'Ry')
world.addJoint(hook, load2, 'Ry')

# Visualisierung

filetyp = "stl_files/%s_01.stl"
world.addVisualisation.File(world, filetyp%'Traeger', name='Traeger')
world.addVisualisation.File(crab, filetyp%'Laufkatze', name='Laufkatze')
world.addVisualisation.File(hook, filetyp%'Haken', name='Haken' )
world.addVisualisation.File(load1, filetyp%'Last', name='Last1' )
world.addVisualisation.File(load2, filetyp%'Last', name='Last2' )

# Steuerung
F = world.addController('F', controlForce, shape=(3, ))
world.addLoad.CmpForce(F, crab, world, name='DrivingForce')

# Bewegungsgleichungen berechnen und Modell darstellen
world.genEquations.Recursive()
#world.show('Laufkatze')

# Get system motion equations
eqns_mo = world.getMotionEquations()

#for eqn in eqns_mo:
#    print str(eqn.lhs) + " = " + str(eqn.rhs)

###############################################################
# by now the following dictionaries have to be set explicitly #
parameters = {'m1' : 54, 'm2' : 2.65, 'm3' : 38, 'm4' : 38, 'g' : 9.81}  #
controller = {'F' : (3,1)}                                    #
###############################################################


######################################################################
# PyTrajectory 
######################################################################

# define function that returns vectorfield
f = sympymbs(eqns_mo, parameters, controller)

# boundary values
a, b = 0.0, 2.0

xa = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#xb = [0.0, np.pi, np.pi, 0.0, 0.0 ,0.0]
xb = [1.0, 0.0, 0.0, 0.0, 0.0 ,0.0]

uab = [0.0, 0.0]

# create trajectory object
T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, g=uab)

# alter some method parameters to increase performance
#T.setParam('kx', 5)
T.setParam('su', 10)
T.setParam('eps', 8e-3)
#T.setParam('use_chains', False)

# run iteration
xt, ut = T.startIteration()

def u_fnc(t):
    if a <= t <= b:
        return ut(t)
    else:
        return ut(b)

dummy_controller.container[0]= u_fnc

world.show('Laufkatze')
#IPS()
