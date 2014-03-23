# coding: utf8

import sys
sys.path.append('..')

from lib.trajectory import Trajectory
import lib.log as log
#from lib.log import IPS
from IPython import embed as IPS

from sympy import cos, sin
import sympy as sp
import numpy as np

import pylab as plt
import os
os.environ['SYMPY_USE_CACHE']='no'

#Acrobot

m=1.0
l=0.5

I=1/3.0*m*l**2
lc = l/2.0
g=9.81

calc=True


def f(x,u):
    x1,x2,x3,x4 = x
    u=u[0]
    d11=m*lc**2+m*(l**2+lc**2+2*l*lc*cos(x1))+2*I
    h1=-m*l*lc*sin(x1)*(x2*(x2+2*x4))
    d12=m*(lc**2+l*lc*cos(x1))+I
    phi1=(m*lc+m*l)*g*cos(x3)+m*lc*g*cos(x1+x3)

    ff = [	x2,
            u,
            x4,
            -1/d11*(h1+phi1+d12*u)]
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
    
    T=Trajectory(f,xa,xb,n=4,m=1)
    
    T.a = 0.0
    T.b = 2.0
    T.sx = 4
    T.su = 10
    T.delta = 2
    T.kx = 5
    T.maxIt  = 5
    T.find_x_chains = True
    T.algo = 'leven'
    T.g = [0,0]
    T.eps = 1e-2
    
    with log.Timer("iteration"):
        T.iteration()
    
    IPS()
    sys.exit()
