# coding: utf8

import sys
sys.path.append('..')

from pytrajectory.trajectory import Trajectory
import pytrajectory.log as log
#from lib.log import IPS
from IPython import embed as IPS

from sympy import cos, sin
import numpy as np
from numpy import pi

#Inverses zweifach Pendel mit partieller Linearisierung [6.2]

calc = True
animate = True

def f(x,u):
	x1, x2, x3, x4, x5, x6 = x
	u, = u
	l1 = 0.7
	l2 = 0.5
	g = 9.81
    
	ff = np.array([   x2,
                        u,
                        x4,
			(1/l1)*(g*sin(x3)+u*cos(x3)),
            		  x6,
			(1/l2)*(g*sin(x5)+u*cos(x5))])
    
	return ff

xa = [0.0 ,0.0 ,pi ,0.0 ,pi ,0.0]
xb = [0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0]


if(calc):
    a, b = (0.0, 2.0)
    su = 10
    uab = [0.0,0.0]
    eps = 8e-2
    
    T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, g=uab, su=su, eps=eps)

    with log.Timer("Iteration()"):
        T.startIteration()



##################################################
# NEW EXPERIMENTAL STUFF
if animate:
    import matplotlib as mpl
    from pytrajectory.utilities import Animation
    
    def draw(xti, image):
        x, phi1, phi2 = xti[0], xti[2], xti[4]
        
        l1 = 0.7
        l2 = 0.5
    
        car_width = 0.05
        car_heigth = 0.02
        pendel_size = 0.015
    
    
        x_car = x
        y_car = 0
    
        x_pendel1 = -l1*sin(phi1)+x_car
        y_pendel1 = l1*cos(phi1)
    
        x_pendel2 = -l2*sin(phi2)+x_car
        y_pendel2 = l2*cos(phi2)
    
        
        # Balls
        sphere1 = mpl.patches.Circle((x_pendel1,y_pendel1),pendel_size,color='k')
        sphere2 = mpl.patches.Circle((x_pendel2,y_pendel2),pendel_size,color='0.3')
        image.patches.append(sphere1)
        image.patches.append(sphere2)
    
        #Car
        car = mpl.patches.Rectangle((x_car-0.5*car_width,y_car-car_heigth),car_width,car_heigth,fill=True,facecolor='0.75',linewidth=2.0)
        image.patches.append(car)
        
        # Gelenk
        gelenk = mpl.patches.Circle((x_car,0),0.005,color='k')
        image.patches.append(gelenk)
        
        # Staebe
        stab1 = mpl.lines.Line2D([x_car,x_pendel1],[y_car,y_pendel1],color='k',zorder=1,linewidth=2.0)
        stab2 = mpl.lines.Line2D([x_car,x_pendel2],[y_car,y_pendel2],color='0.3',zorder=1,linewidth=2.0)
        image.lines.append(stab1)
        image.lines.append(stab2)
        
        return image
    
    A = Animation(drawfnc=draw, simdata=T.sim)
    A.set_limits(xlim=(-1.0,1.1), ylim=(-0.8,0.8))
    
    A.animate()
    A.save('ex4.gif')

