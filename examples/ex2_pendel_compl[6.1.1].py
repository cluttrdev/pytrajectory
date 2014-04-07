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
animate = False

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

if(calc):
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
            
            self.ax.set_xlim(-0.3,0.8);
            self.ax.set_ylim(-0.1,0.6);
            self.ax.set_yticks([])
            self.ax.set_xticks([])
            self.ax.set_position([0.01,0.01,0.98,0.98]);
            self.ax.set_frame_on(True);
            self.ax.set_aspect('equal')
            self.ax.set_axis_bgcolor('w');
        
            self.image=0
    
        def draw(self,x,phi,frame,image=0):
            L=0.5
        
            car_width=0.05
            car_heigth = 0.02
            pendel_size = 0.015
        
        
            x_car=x
            y_car=0
        
            x_pendel=-L*sin(phi)+x_car
            y_pendel= L*cos(phi)
        
            #Init
            if (image==0):
              image=struct()
        
            #update
            else:
                image.sphere.remove()
                image.stab.remove()
                image.car.remove()
                image.gelenk.remove()
        
            #Stab
            image.stab=self.ax.add_line(mpl.lines.Line2D([x_car,x_pendel],[y_car,y_pendel],color='k',zorder=0,linewidth=2.0))
        
            #Ball
            image.sphere=mpl.patches.Circle((x_pendel,y_pendel),pendel_size,color='k')
            self.ax.add_patch(image.sphere)
        
        
            #Car
            image.car=mpl.patches.Rectangle((x_car-0.5*car_width,y_car-car_heigth),car_width,car_heigth,fill=True,facecolor='0.75',linewidth=2.0)
            self.ax.add_patch(image.car)
            #IPS()
            image.gelenk=mpl.patches.Circle((x_car,0),0.005,color='k')
            self.ax.add_patch(image.gelenk)
            #self.ax.annotate(frame, xy=(x_pendel, y_pendel), xytext=(x_pendel+0.02, y_pendel))
        
            #txt = plt.text(x_pendel+0.05,y_pendel,frame)
        
            self.image = image
        
        
            plt.draw()
    
    
    t = T.sim[0]
    xt = T.sim[1]
    
    TT = t[-1] - t[0]
    
    pics = 40
    
    tt = np.linspace(0,(len(t)-1),pics+1,endpoint=True)
    
    M = Modell()
    
    
    def animate(frame):
        i = tt[frame]
        print frame
        M.draw(xt[i,0],xt[i,2],str(round(t[i],2))+'s',image=M.image)
        #sleep(TT/float(pics))
    
    anim = animation.FuncAnimation(M.fig, animate, 
                                   frames=pics, interval=1, blit=True)
    
    
    anim.save('ex2.mp4', fps=20)
    
