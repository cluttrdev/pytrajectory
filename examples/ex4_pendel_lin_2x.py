# coding: utf8

import sys
sys.path.append('..')

from lib.trajectory import Trajectory
import lib.log as log
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
	x1,x2,x3,x4,x5,x6 = x
	u=u[0]
	l1=0.7
	l2=0.5
	g=9.81
	ff = np.array([	x2,
			u,
			x4,
			(1/l1)*(g*sin(x3)+u*cos(x3)),
			x6,
			(1/l2)*(g*sin(x5)+u*cos(x5))])
	return ff

xa=[0.0 ,0.0 ,pi ,0.0 ,pi ,0.0]
xb=[0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0]


if(calc):

    T=Trajectory(f,xa,xb,n=6,m=1)
    
    T.a = 0.0
    T.b = 2.0
    T.sx = 4
    T.su = 10
    T.delta = 2
    T.kx = 2
    T.maxIt  = 6
    T.find_x_chains = True
    T.algo = 'leven'
    #T.gamma = [0.0,0.0]
    T.eps = 8e-2

    with log.Timer("iteration()"):
        T.iteration()
    IPS()
