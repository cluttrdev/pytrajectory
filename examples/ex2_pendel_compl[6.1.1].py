# coding: utf8

import sys
sys.path.append('..')

from lib.trajectory import Trajectory
import lib.log as log
#from lib.log import IPS
from IPython import embed as IPS

from sympy import cos, sin
import numpy as np
import os
os.environ['SYMPY_USE_CACHE']='no'

#Inverses Pendel ohne partielle Linearisierung [6.1.1]

calc=True

def f(x,u):
    x1,x2,x3,x4  =x
    u=u[0]
    l=0.5
    g=9.81
    M=1.0
    m=0.1
    s = sin(x3)
    c = cos(x3)
    ff = np.array([   x2,
            m*s*(-l*x4**2+g*c)/(M+m*s**2)+1/(M+m*s**2)*u,
            x4,
            s*(-m*l*x4**2*c+g*(M+m))/(M*l+m*l*s**2)+c/(M*l+l*m*s**2)*u
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

    T=Trajectory(f,xa,xb,n=4,m=1)
    
    T.a=0.0
    T.b=1.0
    T.sx=4
    T.su=3
    T.delta = 2
    T.kx = 3
    T.maxIt  = 5
    T.find_x_chains = True
    T.algo = 'leven'
    T.g = [0,0]
    T.eps = 8e-2

    with log.Timer("iteration()"):
        T.iteration()
    #T.plot()
    IPS()
