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

calc = True
animate = True

def f(x,u):
    x1, x2, x3, x4 = x
    u1, = u
    
    l = 0.5     # length of the pendulum rod
    g = 9.81    # gravitational acceleration
    M = 1.0     # mass of the cart
    m = 0.1     # mass of the pendulum
    
    s = sin(x3)
    c = cos(x3)
    
    ff = np.array([                     x2,
                   m*s*(-l*x4**2+g*c)/(M+m*s**2)+1/(M+m*s**2)*u1,
                                        x4,
                s*(-m*l*x4**2*c+g*(M+m))/(M*l+m*l*s**2)+c/(M*l+l*m*s**2)*u1
                ])
    return ff


xa = [  0.0,
        0.0,
        0.0,
        0.0]

xb = [  0.5,
        0.0,
        0.0,
        0.0]

if calc:
    a, b = (0.0, 1.0)
    sx = 4
    su = 3
    kx = 3
    maxIt  = 5
    use_chains = False
    g = [0,0]
    eps = 8e-2
    
    T = Trajectory(f,a=a, b=b, xa=xa, xb=xb, sx=sx, su=su, kx=kx, maxIt=maxIt,
                   g=g, eps=eps, use_chains=use_chains)
    
    with log.Timer("Iteration()"):
        T.startIteration()


##################################################
# NEW EXPERIMENTAL STUFF
if animate:
    import numpy as np
    import matplotlib as mpl
    from pytrajectory.utilities import Animation
    
    def draw(xti, image):
        x = xti[0]
        phi = xti[2]
        
        L = 0.5
    
        car_width = 0.05
        car_heigth = 0.02
        pendel_size = 0.015
    
        x_car = x
        y_car = 0
    
        x_pendel =-L*sin(phi)+x_car
        y_pendel = L*cos(phi)
    
        # Stab
        stab = mpl.lines.Line2D([x_car,x_pendel],[y_car,y_pendel],color='k',zorder=0,linewidth=2.0)
        image.lines.append(stab)
    
        # Ball
        sphere = mpl.patches.Circle((x_pendel,y_pendel),pendel_size,color='k')
        image.patches.append(sphere)
    
        # Car
        car = mpl.patches.Rectangle((x_car-0.5*car_width,y_car-car_heigth),car_width,car_heigth,
                                    fill=True,facecolor='0.75',linewidth=2.0)
        image.patches.append(car)
        
        # Gelenk
        gelenk = mpl.patches.Circle((x_car,0),0.005,color='k')
        image.patches.append(gelenk)
        
        return image
    
    A = Animation(drawfnc=draw, simdata=T.sim,
                    plotsys=[[0, 'x'], [2, 'phi']], plotinputs=[[0, 'F']])
    A.set_limits(xlim=(-0.3,0.8), ylim=(-0.1,0.6))
    
    A.animate()
    A.save('ex2.gif')
