import sys
sys.path.append('..')

from pytrajectory.trajectory import Trajectory
import pytrajectory.log as log
#from pytrajectory.log import IPS
from IPython import embed as IPS

import numpy as np
from sympy import cos, sin
from numpy import pi


# partiell linearisiertes inverses Pendel [6.1.3]

calc = True
animate = False

def f(x,u):
    x1, x2, x3, x4 = x
    u1, = u
    l = 0.5
    g = 9.81
    ff = np.array([     x2,
                        u1,
                        x4,
                    (1/l)*(g*sin(x3)+u1*cos(x3))])
    return ff

xa = [  0.0,
        0.0,
        pi,
        0.0]

xb = [  0.0,
        0.0,
        0.0,
        0.0]

if calc:
    a = 0.0
    b = 2.0
    sx = 5
    su = 5
    kx = 5
    maxIt  = 5
    _g = [0,0]
    eps = 0.05
    use_chains = False

    T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, sx=sx, su=su, kx=kx,
                    maxIt=maxIt, g=_g, eps=eps, use_chains=use_chains)

    with log.Timer("startIteration"):
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
        
        L=0.5
        
        car_width =0.05
        car_heigth = 0.02
        pendel_size = 0.015
        
        x_car = x
        y_car = 0
        
        x_pendel =-L*sin(phi)+x_car
        y_pendel = L*cos(phi)
        
        # build image
        sphere = mpl.patches.Circle((x_pendel,y_pendel),pendel_size,color='k')
        image.patches.append(sphere)
        
        car = mpl.patches.Rectangle((x_car-0.5*car_width,y_car-car_heigth),car_width,car_heigth,
                                    fill=True,facecolor='0.75',linewidth=2.0)
        image.patches.append(car)
        
        gelenk = mpl.patches.Circle((x_car,0),0.005,color='k')
        image.patches.append(gelenk)
        
        stab = mpl.lines.Line2D([x_car,x_pendel],[y_car,y_pendel],color='k',zorder=1,linewidth=2.0)
        image.lines.append(stab)
        
        return image
    
    A = Animation(drawfnc=draw, simdata=T.sim,
                    plotsys=[[0, 'x'], [2, 'phi']], plotinputs=[[0, 'F']])
    A.set_limits(xlim=(-1.2,0.3), ylim=(-0.6,0.6))
    
    A.animate()
    A.save('ex1.mp4')

