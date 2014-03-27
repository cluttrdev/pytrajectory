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
    x1,x2,x3,x4,x5,x6 = x
    u1,u2 = u
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
    ff = [   x2,
            -s/M*(u1+u2) + c/M*(u1-u2)*sa,
            x4,
            -g+c/M*(u1+u2) +s/M*(u1-u2)*sa ,
            x6,
            1/J*(u1-u2)*(l*ca+h*sa)]
    return ff 


xa=[0.0,0.0,0.0,0.0,0.0,0.0]
xb=[10.0,0.0,5.0,0.0,0.0,0.0]

if(calc):

    T=Trajectory(f,xa,xb)#,n=6,m=2)
    
    T.a=0.0
    T.b=3.0
    T.sx=4
    T.su=3
    T.delta = 2
    T.kx = 5
    T.maxIt  = 6
    T.find_x_chains = False
    T.algo = 'leven'
    T.g = [0.5*9.81*50.0/(cos(5/360.0*2*pi)),0.5*9.81*50.0/(cos(5/360.0*2*pi))]
    T.eps = 1e-2

    with log.Timer("iteration"):
        T.iteration()
    IPS()
