# coding: utf8
from __future__ import absolute_import

from sympy import cos, sin
import sympy as sp
import numpy as np
from lib.trajectory import Trajectory
from lib.tools import setAxLinesBW, setFigLinesBW
from ipHelp import IPS, ST, ip_syshook, dirsearch, sys
import pylab as plt
import pickle
import os
os.environ['SYMPY_USE_CACHE']='no'

#import cProfile zuzm Debugen

#linearisiertes inverses Pendel [6.1.3]

calc=False

def f(x,u):
	x1,x2,x3,x4 = x
	u=u[0]
	l=0.5
	g=9.81
	ff = np.array([     x2,
                        u,
			            x4,
			            (1/l)*(g*sin(x3)+u*cos(x3))])
	return ff

xa=[	0.0,
		0.0,
		np.pi,
		0.0]

xb=[	0.0,
		0.0,
		0.0,
		0.0]
#T=Trajectory(f,xa,xb,b=1.0,n=4,m=1,lambda_x=5,kappa=2,v_max=5,solver='newton_algopy',lambda_u=5,mul=1,u_is_zero=True)

#T=Trajectory(f,xa,xb,a=0.0,b=1.0,n=4,m=1,lambda_x=5,kappa=3,lambda_u=1,mul=1,v_max=3,solver='newton_algopy')

if(calc):

    T=Trajectory(f,xa,xb,n=4,m=1)
    
    T.a=0.0
    T.b=2.0
    T.lambda_x=5
    T.lambda_u=5
    T.mul = 2
    T.kappa = 3
    T.v_max  = 4
    T.find_integ = True
    T.solver = 'leven_algopy'
    T.gamma = [0,0]
    T.epsilon_tol = 1e-2


    T.iteration()
    IPS()
    #T.plot()

    save={}

    save['xa']=T.xa
    save['xb']=T.xb
    save['splines']=T.splines
    save['x_sym']=T.x_sym
    save['u_sym']=T.u_sym
    save['A']=T.A
    save['n']=T.n
    save['m']=T.m
    save['solver']=T.solver
    save['delta']=T.delta
    save['lambda_x_start']=T.lambda_x_start
    save['lambda_x']=T.lambda_x
    save['lambda_u']=T.lambda_u
    save['a']=T.a
    save['b']=T.b
    save['v_max']=T.v_max
    save['kappa']=T.kappa
    save['epsilon_tol']=T.epsilon_tol
    save['tol']=T.tol
    save['find_integ']=T.find_integ
    save['gamma']=T.gamma
    save['l2']=T.l2
    save['H']=T.H

#die wären wichtig
#save['x_func']=T.x_func
#save['u_func']=T.u_func

    output = open('ex1_pendel_aufschwingen.pkl', 'wb')

# Pickle dictionary using protocol 0.
    pickle.dump(save, output)

    output.close()


TT=pickle.load(open('ex1_pendel_aufschwingen.pkl', 'rb'))
#IPS()

A=TT['A']
xa=TT['xa']
xb=TT['xb']
l2=TT['l2']
x_sym=TT['x_sym']
u_sym=TT['u_sym']
H=TT['H']

t=A[0]
xt = A[1]
ut= A[2]


plt.rcParams['figure.subplot.bottom']=.2
plt.rcParams['figure.subplot.top']= .95
plt.rcParams['figure.subplot.left']=.13
plt.rcParams['figure.subplot.right']=.95

plt.rcParams['font.size']=16

plt.rcParams['legend.fontsize']=16
plt.rc('text', usetex=True)
        

plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16
plt.rcParams['legend.fontsize']=20

plt.rcParams['axes.titlesize']=26
plt.rcParams['axes.labelsize']=26


plt.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='8'

mm = 1./25.4 #mm to inch
scale = 3
fs = [100*mm*scale, 60*mm*scale]

fff=plt.figure(figsize=fs, dpi=80)


a=2
b=3
PP=1

#x1
plt.subplot(a,b,PP)
lines = plt.plot(t,xt[:,PP-1])     
plt.xlabel(r'$t/s$')
plt.title(r'$x_1(t)/m$')  
PP+=1

#x2
plt.subplot(a,b,PP)
lines = plt.plot(t,xt[:,PP-1])     
plt.xlabel(r'$t/s$')
plt.title(r'$x_2(t)/\frac{m}{s}$')  
PP+=1 

#IPS()
#x3
plt.subplot(a,b,PP)
 
plt.xlabel(r'$t/s$')
y_pi   = xt[:,PP-1]#/np.pi
lines = plt.plot(t,y_pi)    
#IPS()
unit   = 1/2.0
y_tick = np.arange(0, 1.0+unit, unit)
y_label = [r"$0$", r"$+\frac{\pi}{2}$",r"$+\pi$"]
plt.yticks(y_tick*np.pi,y_label)
plt.title(r'$x_3(t)/rad$')  
PP+=1  

#x4
plt.subplot(a,b,PP)
lines = plt.plot(t,xt[:,PP-1])     
plt.xlabel(r'$t/s$')
plt.title(r'$x_4(t)/\frac{rad}{s}$')  
PP+=1 

#IPS()
#u
plt.subplot(a,b,PP)
PP+=1
lines = plt.plot(t,ut[:,0])     
plt.xlabel(r'$t/s$')
plt.title(r'$\tilde u(t)/\frac{m}{s^2}$')  

        
for hh in H:

    plt.subplot(a,b,PP)
    PP+=1
    lines = plt.plot(t,H[hh])     
    plt.xlabel(r'$t/s$')
    plt.title(r'$H_'+str(hh+1)+'(t)$')  

setFigLinesBW(fff)

plt.tight_layout()

print 'Ending up with:'
for i,xx in enumerate(x_sym):
    print str(xx),':', xt[-1:][0][i]

print 'Should be:'
for i,xx in enumerate(x_sym):
    print str(xx),':', xb[xx]

print 'Difference:'
for i,xx in enumerate(x_sym):
    print str(xx),':',xb[xx]-xt[-1:][0][i]

plt.show()


IPS()
