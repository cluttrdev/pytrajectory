# coding: utf8

from sympy import cos, sin
import sympy as sp
import numpy as np
from lib.trajectory import Trajectory
from ipHelp import IPS, ST, ip_syshook, dirsearch, sys
from numpy import pi
import pylab as plt
from lib.tools import setAxLinesBW, setFigLinesBW
import pickle
import os
os.environ['SYMPY_USE_CACHE']='no'

#Inverses zweifach Pendel mit partieller Linearisierung [6.2]

calc=False

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
    
    T.a=0.0
    T.b=2.0
    T.lambda_x=4
    T.lambda_u=10
    T.delta = 2
    T.kappa = 2
    T.v_max  = 6
    T.find_integ = True
    T.algo = 'leven'
    #T.gamma = [0.0,0.0]
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
    save['algo']=T.algo
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

    output = open('ex4_2xpendel_aufschwingen2.pkl', 'wb')

# Pickle dictionary using protocol 0.
    pickle.dump(save, output)

    output.close()

    exit()


TT=pickle.load(open('ex4_2xpendel_aufschwingen2.pkl', 'rb'))
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
scale = 4
fs = [100*mm*scale, 80*mm*scale]

fff=plt.figure(figsize=fs, dpi=80)


a=3
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

#x5
plt.subplot(a,b,PP)
 
plt.xlabel(r'$t/s$')
y_pi   = xt[:,PP-1]#/np.pi
lines = plt.plot(t,y_pi)    
#IPS()
unit   = 1/2.0
y_tick = np.arange(0, 1.0+unit, unit)
y_label = [r"$0$", r"$+\frac{\pi}{2}$",r"$+\pi$"]
plt.yticks(y_tick*np.pi,y_label)
plt.title(r'$x_5(t)/rad$')  
PP+=1  

#x6
plt.subplot(a,b,PP)
lines = plt.plot(t,xt[:,PP-1])     
plt.xlabel(r'$t/s$')
plt.title(r'$x_6(t)/\frac{rad}{s}$')  
PP+=1 

#IPS()
#u
plt.subplot(a,b,PP)
PP+=1
lines = plt.plot(t,ut[:,0])     
plt.xlabel(r'$t/s$')
plt.title(r'$\tilde u(t)/\frac{m}{s^2}$')  

        

plt.subplot(a,b,PP)
PP+=1
lines = plt.plot(t,H[3])     
plt.xlabel(r'$t/s$')
plt.title(r'$H_4(t)$')  

plt.subplot(a,b,PP)
PP+=1
lines = plt.plot(t,H[5])     
plt.xlabel(r'$t/s$')
plt.title(r'$H_6(t)$')  

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