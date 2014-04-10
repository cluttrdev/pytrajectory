import sys
sys.path.append('..')

from pytrajectory.trajectory import Trajectory
import pytrajectory.log as log
#from lib.log import IPS
from IPython import embed as IPS

from sympy import cos, sin
import numpy as np

#Acrobot

m = 1.0
l = 0.5

I = 1/3.0*m*l**2
lc = l/2.0
g = 9.81

calc = True
animate = True


def f(x,u):
    x1, x2, x3, x4 = x
    u1, = u
    
    d11 = m*lc**2+m*(l**2+lc**2+2*l*lc*cos(x1))+2*I
    h1 = -m*l*lc*sin(x1)*(x2*(x2+2*x4))
    d12 = m*(lc**2+l*lc*cos(x1))+I
    phi1 = (m*lc+m*l)*g*cos(x3)+m*lc*g*cos(x1+x3)

    ff = np.array([	    x2,
                        u1,
                        x4,
                -1/d11*(h1+phi1+d12*u1)
                ])
    
    return ff

#Aufschwingen

xa = [  0.0,
        0.0,
        3/2.0*np.pi,
        0.0]

xb = [  0.0,
        0.0,
        1/2.0*np.pi,
        0.0]

if calc:
    a, b = (0.0, 2.0)
    sx = 4
    su = 10
    use_chains = True
    _g = [0,0]
    eps = 1e-2
    
    T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, sx=sx, su=su,
                   eps=eps, g=_g, use_chains=use_chains)
    
    with log.Timer("Iteration"):
        T.startIteration()


##################################################
# NEW EXPERIMENTAL STUFF
if animate:
    import matplotlib as mpl
    from pytrajectory.utilities import Animation
    
    def draw(xti, image):
        phi1, phi2 = xti[0], xti[2]
        
        L=0.5
        
        x1 = L*cos(phi2)
        y1 = L*sin(phi2)
        
        x2 = x1+L*cos(phi2+phi1)
        y2 = y1+L*sin(phi2+phi1)
        
        #Staebe
        stab1 = mpl.lines.Line2D([0,x1],[0,y1],color='k',zorder=0,linewidth=2.0)
        stab2 = mpl.lines.Line2D([x1,x2],[y1,y2],color='0.3',zorder=0,linewidth=2.0)
        image.lines.append(stab1)
        image.lines.append(stab2)
        
        #Balls
        sphere1 = mpl.patches.Circle((x1,y1),0.01,color='k')
        sphere2 = mpl.patches.Circle((0,0),0.01,color='k')
        image.patches.append(sphere1)
        image.patches.append(sphere2)
        
        return image
    
    A = Animation(drawfnc=draw, simdata=T.sim, 
                  plotsys=[[0,'phi1'],[2,'phi2']], plotinputs=[[0, 'u1']])
    A.set_limits(xlim=(-1.1,1.1), ylim=(-1.1,1.1))
    
    A.animate()
    A.save('ex6.mp4')
