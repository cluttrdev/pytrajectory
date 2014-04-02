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

#Inverses Pendel ohne partielle Linearisierung [6.1.1]

calc=True

def f(x,u):
    x1, x2, x3, x4 = x
    u1, = u
    l = 0.5
    g = 9.81
    M = 1.0
    m = 0.1
    s = sin(x3)
    c = cos(x3)
    ff = np.array([                     x2,
                   m*s*(-l*x4**2+g*c)/(M+m*s**2)+1/(M+m*s**2)*u1,
                                        x4,
                s*(-m*l*x4**2*c+g*(M+m))/(M*l+m*l*s**2)+c/(M*l+l*m*s**2)*u1
                ])
    return ff


xa=[    0.0,
        0.0,
        0.0,
        0.0]

xb=[    0.5,
        0.0,
        0.0,
        0.0]

if(calc):
    a, b = (0.0, 1.0)
    sx = 4
    su = 3
    kx = 3
    maxIt  = 5
    use_chains = True # if set to False the method diverges at 36 spline parts
    g = [0,0]
    eps = 8e-2
    
    T = Trajectory(f,a=a, b=b, xa=xa, xb=xb, sx=sx, su=su, kx=kx, maxIt=maxIt,
                   g=g, eps=eps, use_chains=use_chains)
    
    with log.Timer("Iteration()"):
        T.startIteration()
