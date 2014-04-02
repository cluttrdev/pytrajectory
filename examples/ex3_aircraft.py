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


calc = True
animate = False

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
    ff = [              x2,
            -s/M*(u1+u2) + c/M*(u1-u2)*sa,
                        x4,
            -g+c/M*(u1+u2) +s/M*(u1-u2)*sa ,
                        x6,
            1/J*(u1-u2)*(l*ca+h*sa)]
    return ff 


xa = [0.0,0.0,0.0,0.0,0.0,0.0]
xb = [10.0,0.0,5.0,0.0,0.0,0.0]

if(calc):
    a, b = (0.0, 3.0)
    sx = 4
    su = 3
    kx = 5
    eps = 1e-2
    g = [0.5*9.81*50.0/(cos(5/360.0*2*pi)),0.5*9.81*50.0/(cos(5/360.0*2*pi))]
    
    use_chains = False
    
    T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, sx=sx, su=su, kx=kx, eps=eps,
                 g=g, use_chains=use_chains)
    
    with log.Timer("Iteration"):
        T.startIteration()


##################################################
# NEW EXPERIMENTAL STUFF
# --> taken from original version
if animate:
    import numpy as np
    import pylab as plt
    import matplotlib as mpl
    from matplotlib import animation
    
    class struct():
        def __init__(self):
            return
    
    class Modell:
        def __init__(self):
            self.fig=plt.figure()
            self.ax=plt.axes()
        
            mng = plt.get_current_fig_manager()
        
            #mng.window.wm_geometry("1000x700+50+50")  
            mng.window.setGeometry(0, 0, 1000, 700)
        
            self.ax.set_xlim(-1,11);
            self.ax.set_ylim(-1,6);
            self.ax.set_yticks([])
            self.ax.set_xticks([])
            self.ax.set_position([0.01,0.01,0.98,0.98]);
            self.ax.set_frame_on(True);
            self.ax.set_aspect('equal')
            self.ax.set_axis_bgcolor('w');
        
            self.image=0
    
        def draw(self,x,y,theta,frame,image=0):
            #Init
            if (image==0):
                image=struct()
            else:
                image.aircraft.remove()
        
            S = np.array( [   [0,     0.3],
                              [-0.1,  0.1],
                              [-0.7,  0],
                              [-0.1,  -0.05],
                              [ 0,    -0.1],
                              [0.1,   -0.05],
                              [ 0.7,  0],
                              [ 0.1,  0.1]])
        
            xx=S[:,0].copy()
            yy=S[:,1].copy()
        
            S[:,0]=xx*cos(theta)-yy*sin(theta)+x
            S[:,1]=yy*cos(theta)+xx*sin(theta)+y
            # IPS()
            # S[:,0]=xx+x
            # S[:,1]=yy+y
        
            #IPS()
            image.aircraft = mpl.patches.Polygon(S, closed=True,facecolor = '0.75')
            self.ax.add_patch(image.aircraft)
        
            #IPS()
        
            #self.ax.annotate(frame, xy=(x_pendel, y_pendel), xytext=(x_pendel+0.02, y_pendel))
        
            #txt = plt.text(x_pendel+0.05,y_pendel,frame)
        
            self.image = image
        
        
            plt.draw()
    
    
    t = T.sim[0]
    xt = T.sim[1]
    
    TT = t[-1] - t[0]
    
    pics = 60
    
    tt = np.linspace(0,(len(t)-1),pics+1,endpoint=True)
    
    M = Modell()
    
    def animate(frame):
        i = tt[frame]
        print frame
        M.draw(xt[i,0],xt[i,2],xt[i,4],str(round(t[i],2))+'s',image=M.image)
        #sleep(T/float(pics))
    
    anim = animation.FuncAnimation(M.fig, animate, 
                                   frames=pics, interval=1, blit=True)
    
    
    anim.save('ex3.mp4', fps=20)
    
