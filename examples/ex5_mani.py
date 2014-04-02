# coding: utf8
import sys
sys.path.append('..')

from pytrajectory.trajectory import Trajectory
import pytrajectory.log as log
#from lib.log import IPS

from sympy import cos, sin
import os

from numpy import pi
os.environ['SYMPY_USE_CACHE']='no'

from IPython import embed as IPS

#unteraktuirter Manipulator

calc = True
animate = False

def f(x,u):
    x1, x2, x3, x4  =x
    u, = u
    e = 0.9
    s = sin(x3)
    c = cos(x3)
    ff = [          x2,
                    u,
                    x4,
            -e*x2**2*s-(1+e*c)*u
        ]
    return ff


xa=[    0.0,
        0.0,
        0.4*pi,
        0.0]

xb=[    0.2*pi,
        0.0,
        0.2*pi,
        0.0]
if(calc):
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
        
            self.ax.set_xlim(-0.1,0.6);
            self.ax.set_ylim(-0.4,0.65);
            self.ax.set_yticks([])
            self.ax.set_xticks([])
            self.ax.set_position([0.01,0.01,0.98,0.98]);
            self.ax.set_frame_on(True);
            self.ax.set_aspect('equal')
            self.ax.set_axis_bgcolor('w');
        
            self.image=0
    
        def draw(self,phi1,phi2,frame,image=0):
            L=0.4
        
            x1=L*cos(phi1)
            y1=L*sin(phi1)
        
            x2=x1+L*cos(phi2+phi1)
            y2=y1+L*sin(phi2+phi1)
        
            #Init
            if (image==0):
                image=struct()
        
            #update
            else:
                image.sphere1.remove()
                image.sphere2.remove()
                image.stab1.remove()
                image.stab2.remove()
        
            #Stab
            image.stab1=self.ax.add_line(mpl.lines.Line2D([0,x1],[0,y1],color='k',zorder=0,linewidth=2.0))
            image.stab2=self.ax.add_line(mpl.lines.Line2D([x1,x2],[y1,y2],color='k',zorder=0,linewidth=2.0))
        
            image.sphere1=mpl.patches.Circle((x1,y1),0.01,color='k')
            self.ax.add_patch(image.sphere1)
            image.sphere2=mpl.patches.Circle((0,0),0.01,color='k')
            self.ax.add_patch(image.sphere2)
        
            #txt = plt.text(x_pendel+0.05,y_pendel,frame)
        
            self.image = image
        
            plt.draw()
    
    
    t = T.sim[0]
    xt = T.sim[1]
    
    TT = t[-1] - t[0]
    
    pics = 80
    
    tt = np.linspace(0,(len(t)-1),pics+1,endpoint=True)
    
    M = Modell()
    
    def animate(frame):
        i = tt[frame]
        print frame
        M.draw(xt[i,0],xt[i,2],str(round(t[i],2))+'s',image=M.image)
        #sleep(TT/float(pics))
    
    anim = animation.FuncAnimation(M.fig, animate, 
                                   frames=pics, interval=1, blit=True)
    
    anim.save('ex5.mp4', fps=20)
    
