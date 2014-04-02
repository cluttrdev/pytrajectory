# coding: utf8

import sys
sys.path.append('..')

from sympy import cos, sin

from pytrajectory.trajectory import Trajectory
from numpy import pi
import pytrajectory.log as log
#from lib.log import IPS
from IPython import embed as IPS

import os
os.environ['SYMPY_USE_CACHE']='no'

#senkrecht startendes Flugzeug [6.3]


calc=True

def f(x,u):
    x1, x2, x3, x4, x5, x6 = x
    u1, u2 = u
    l = 1.0
    h = 0.1

    g = 9.81
    M = 50.0
    J = 25.0

    alpha = 5/360.0*2*pi

    sa = sin(alpha)
    ca = cos(alpha)

    s = sin(x5)
    c = cos(x5)
    ff = [              x2,
            -s/M*(u1+u2) + c/M*(u1-u2)*sa,
                        x4,
            -g+c/M*(u1+u2) +s/M*(u1-u2)*sa ,
                        x6,
            1/J*(u1-u2)*(l*ca+h*sa)]
    return ff 


xa = [0.0,0.0,0.0,0.0,0.0,0.0]
xb = [10.0,0.0,5.0,0.0,0.0,0.0]

if(calc):
    a, b = (0.0, 3.0)
    sx = 4
    su = 3
    kx = 5
    eps = 1e-2
    g = [0.5*9.81*50.0/(cos(5/360.0*2*pi)),0.5*9.81*50.0/(cos(5/360.0*2*pi))]
    
    use_chains = False
    
    T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, sx=sx, su=su, kx=kx, eps=eps,
                 g=g, use_chains=use_chains)
    
    with log.Timer("Iteration"):
        T.startIteration()
