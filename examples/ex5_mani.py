# coding: utf8
import sys
sys.path.append('..')

from pytrajectory.trajectory import Trajectory
import pytrajectory.log as log
#from lib.log import IPS

import numpy as np
from sympy import cos, sin
from numpy import pi

from IPython import embed as IPS

#unteraktuirter Manipulator

calc = True
animate = False

def f(x,u):
    x1, x2, x3, x4  = x
    u1, = u
    
    e = 0.9
    s = sin(x3)
    c = cos(x3)
    
    ff = np.array([         x2,
                            u1,
                            x4,
                    -e*x2**2*s-(1+e*c)*u1
                    ])
    
    return ff


xa = [  0.0,
        0.0,
        0.4*pi,
        0.0]

xb = [  0.2*pi,
        0.0,
        0.2*pi,
        0.0]

if calc:
    a, b = (0.0, 1.8)
    sx = 5
    su = 20
    kx = 3
    use_chains = True
    g = [0,0]
    eps = 1e-2
    
    T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, sx=sx, su=su, kx=kx,
                   eps=eps, g=g, use_chains=use_chains)
    
    with log.Timer("Iteration()"):
        T.startIteration()


##################################################
# NEW EXPERIMENTAL STUFF
# --> taken from original version
if animate:
    import matplotlib as mpl
    from pytrajectory.utilities import Animation
    
    def draw(xti, image):
        phi1, phi2 = xti[0], xti[2]
        
        L =0.4
        
        x1 = L*cos(phi1)
        y1 = L*sin(phi1)
        
        x2 = x1+L*cos(phi2+phi1)
        y2 = y1+L*sin(phi2+phi1)
        
        # Stab
        stab1 = mpl.lines.Line2D([0,x1],[0,y1],color='k',zorder=0,linewidth=2.0)
        stab2 = mpl.lines.Line2D([x1,x2],[y1,y2],color='k',zorder=0,linewidth=2.0)
        image.lines.append(stab1)
        image.lines.append(stab2)
        
        # Balls
        sphere1 = mpl.patches.Circle((x1,y1),0.01,color='k')
        sphere2 = mpl.patches.Circle((0,0),0.01,color='k')
        image.patches.append(sphere1)
        image.patches.append(sphere2)
        
        return image
    
    A = Animation(drawfnc=draw, simdata=T.sim)
    A.set_limits(xlim= (-0.1,0.6), ylim=(-0.4,0.65))
    
    A.animate()
    A.save('ex5.mp4')
