# coding: utf8
import sys
sys.path.append('..')

from pytrajectory.trajectory import Trajectory
import pytrajectory.log as log
#from lib.log import IPS

from sympy import cos, sin
import os

from numpy import pi
os.environ['SYMPY_USE_CACHE']='no'

from IPython import embed as IPS

#unteraktuirter Manipulator

calc=True

def f(x,u):
    x1, x2, x3, x4  =x
    u, = u
    e = 0.9
    s = sin(x3)
    c = cos(x3)
    ff = [          x2,
                    u,
                    x4,
            -e*x2**2*s-(1+e*c)*u
        ]
    return ff


xa=[    0.0,
        0.0,
        0.4*pi,
        0.0]

xb=[    0.2*pi,
        0.0,
        0.2*pi,
        0.0]
if(calc):
    a, b = (0.0, 1.8)
    sx = 5
    su = 20
    kx = 3
    use_chains = True
    g = [0,0]
    eps = 1e-2
    
    T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, sx=sx, su=su, kx=kx,
                   eps=eps, g=g, use_chains=use_chains)
    
    with log.Timer("Iteration()"):
        T.startIteration()
    IPS()
