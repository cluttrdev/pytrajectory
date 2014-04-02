# coding: utf8

import sys
sys.path.append('..')

from pytrajectory.trajectory import Trajectory
import pytrajectory.log as log
#from lib.log import IPS
from IPython import embed as IPS

from sympy import cos, sin
import numpy as np
from numpy import pi

import os
os.environ['SYMPY_USE_CACHE']='no'

#Inverses zweifach Pendel mit partieller Linearisierung [6.2]

calc=True

def f(x,u):
	x1, x2, x3, x4, x5, x6 = x
	u, = u
	l1 = 0.7
	l2 = 0.5
	g = 9.81
	ff = np.array([   x2,
                        u,
                        x4,
			(1/l1)*(g*sin(x3)+u*cos(x3)),
            		  x6,
			(1/l2)*(g*sin(x5)+u*cos(x5))])
	return ff

xa = [0.0 ,0.0 ,pi ,0.0 ,pi ,0.0]
xb = [0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0]


if(calc):
    a, b = (0.0, 2.0)
    sx = 4
    su = 10
    kx = 2
    use_chains = True
    #g = [0.0,0.0]
    eps = 8e-2
    
    T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, sx=sx, su=su, kx=kx, eps=eps,
                   use_chains=use_chains)

    with log.Timer("Iteration()"):
        T.startIteration()
