# coding: utf8

import sys
sys.path.append('..')

from sympy import cos, sin

from pytrajectory.trajectory import Trajectory
import numpy as np
from numpy import pi
import pytrajectory.log as log
#from lib.log import IPS
from IPython import embed as IPS

#senkrecht startendes Flugzeug [6.3]


calc = True
animate = 1

def f(x,u):
    x1, x2, x3, x4, x5, x6 = x
    u1, u2 = u
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
    
    ff = np.array([             x2,
                    -s/M*(u1+u2) + c/M*(u1-u2)*sa,
                                x4,
                    -g+c/M*(u1+u2) +s/M*(u1-u2)*sa ,
                                x6,
                    1/J*(u1-u2)*(l*ca+h*sa)])
    
    return ff 


xa = [0.0,0.0,0.0,0.0,0.0,0.0]
xb = [10.0,0.0,5.0,0.0,0.0,0.0]

if(calc):
    a, b = (0.0, 3.0)
    kx = 5
    uab = [0.5*9.81*50.0/(cos(5/360.0*2*pi)),0.5*9.81*50.0/(cos(5/360.0*2*pi))]
    
    use_chains = False
    
    T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, g= uab, kx=kx, use_chains=use_chains)
    
    with log.Timer("Iteration"):
        T.startIteration()


##################################################
# NEW EXPERIMENTAL STUFF
if animate:
    import matplotlib as mpl
    from pytrajectory.utilities import Animation
    
    def draw(xti, image):
        x, y, theta = xti[0], xti[2], xti[4]
        
        S = np.array( [   [0,     0.3],
                          [-0.1,  0.1],
                          [-0.7,  0],
                          [-0.1,  -0.05],
                          [ 0,    -0.1],
                          [0.1,   -0.05],
                          [ 0.7,  0],
                          [ 0.1,  0.1]])
    
        xx = S[:,0].copy()
        yy = S[:,1].copy()
    
        S[:,0] = xx*cos(theta)-yy*sin(theta)+x
        S[:,1] = yy*cos(theta)+xx*sin(theta)+y
    
        aircraft = mpl.patches.Polygon(S, closed=True, facecolor='0.75')
        image.patches.append(aircraft)
        
        return image
    
    A = Animation(drawfnc=draw, simdata=T.sim)
    A.set_limits(xlim=(-1,11), ylim=(-1,7))
    
    A.animate()
    A.save('ex3.gif')
