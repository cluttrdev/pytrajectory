# coding: utf8
import sys
sys.path.append('..')

from lib.trajectory import Trajectory
import lib.log as log
#from lib.log import IPS

from sympy import cos, sin
import sympy as sp
import numpy as np
import os

from numpy import pi
os.environ['SYMPY_USE_CACHE']='no'

from IPython import embed as IPS

#unteraktuirter Manipulator

calc=True

def f(x,u):
    x1,x2,x3,x4  =x
    u=u[0]
    e=0.9
    s = sin(x3)
    c = cos(x3)
    ff = [  x2,
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

    T=Trajectory(f,xa,xb,n=4,m=1)
    
    T.a = 0.0
    T.b = 1.8
    T.sx = 5
    T.su = 20
    T.delta = 2
    T.kx = 3
    T.maxIt  = 5
    T.find_x_chains = True
    T.algo = 'leven'
    T.g = [0,0]
    T.eps = 1e-2

    with log.Timer("iteration()"):
        T.iteration()
    #T.plot()
    IPS()
