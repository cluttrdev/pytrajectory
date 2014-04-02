# coding: utf8

import sys
sys.path.append('..')

from pytrajectory.trajectory import Trajectory
import pytrajectory.log as log
#from lib.log import IPS
from IPython import embed as IPS

from sympy import cos, sin
import numpy as np

import os
os.environ['SYMPY_USE_CACHE']='no'

#Acrobot

m = 1.0
l = 0.5

I = 1/3.0*m*l**2
lc = l/2.0
g = 9.81

calc = True


def f(x,u):
    x1, x2, x3, x4 = x
    u1, = u
    d11 = m*lc**2+m*(l**2+lc**2+2*l*lc*cos(x1))+2*I
    h1 = -m*l*lc*sin(x1)*(x2*(x2+2*x4))
    d12 = m*(lc**2+l*lc*cos(x1))+I
    phi1 = (m*lc+m*l)*g*cos(x3)+m*lc*g*cos(x1+x3)

    ff = [	x2,
            u1,
            x4,
            -1/d11*(h1+phi1+d12*u1)]
    return ff

#Aufschwingen

xa=[	0.0,
	0.0,
	3/2.0*np.pi,
	0.0]

xb=[	0.0,
	0.0,
	1/2.0*np.pi,
	0.0]

if(calc):
    a, b = (0.0, 2.0)
    sx = 4
    su = 10
    use_chains = False # --> if set to True, method will fail because of singular matrix...???
    _g = [0,0]
    eps = 1e-2
    
    T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, sx=sx, su=su,
                   eps=eps, g=_g, use_chains=use_chains)
    
    with log.Timer("Iteration"):
        T.startIteration()
