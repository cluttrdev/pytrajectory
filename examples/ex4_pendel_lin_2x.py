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

import os
os.environ['SYMPY_USE_CACHE']='no'

#Inverses zweifach Pendel mit partieller Linearisierung [6.2]

calc = True
animate = False

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
    sx = 4
    su = 10
    kx = 2
    use_chains = True
    #g = [0.0,0.0]
    eps = 8e-2
    
    T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, sx=sx, su=su, kx=kx, eps=eps,
                   use_chains=use_chains)

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
        
            self.ax.set_xlim(-1.0,0.8);
            self.ax.set_ylim(-0.8,0.8);
            self.ax.set_yticks([])
            self.ax.set_xticks([])
            self.ax.set_position([0.01,0.01,0.98,0.98]);
            self.ax.set_frame_on(True);
            self.ax.set_aspect('equal')
            self.ax.set_axis_bgcolor('w');
        
            self.image=0
    
        def draw(self,x,phi1,phi2,frame,image=0):
            l1=0.7
            l2=0.5
        
            car_width=0.05
            car_heigth = 0.02
            pendel_size = 0.015
        
        
            x_car=x
            y_car=0
        
            x_pendel1=-l1*sin(phi1)+x_car
            y_pendel1= l1*cos(phi1)
        
            x_pendel2=-l2*sin(phi2)+x_car
            y_pendel2= l2*cos(phi2)
        
            #Init
            if (image==0):
              image=struct()
        
            #update
            else:
                image.sphere1.remove()
                image.sphere2.remove()
                image.stab1.remove()
                image.stab2.remove()
                image.car.remove()
                image.gelenk.remove()
        
            
            #Ball
            image.sphere1=mpl.patches.Circle((x_pendel1,y_pendel1),pendel_size,color='k')
            self.ax.add_patch(image.sphere1)
            image.sphere2=mpl.patches.Circle((x_pendel2,y_pendel2),pendel_size,color='0.3')
            self.ax.add_patch(image.sphere2)
        
            #Car
            image.car=mpl.patches.Rectangle((x_car-0.5*car_width,y_car-car_heigth),car_width,car_heigth,fill=True,facecolor='0.75',linewidth=2.0)
            self.ax.add_patch(image.car)
            #IPS()
            image.gelenk=mpl.patches.Circle((x_car,0),0.005,color='k')
            self.ax.add_patch(image.gelenk)
            #self.ax.annotate(frame, xy=(x_pendel, y_pendel), xytext=(x_pendel+0.02, y_pendel))
            #Stab
            image.stab1=self.ax.add_line(mpl.lines.Line2D([x_car,x_pendel1],[y_car,y_pendel1],color='k',zorder=1,linewidth=2.0))
            image.stab2=self.ax.add_line(mpl.lines.Line2D([x_car,x_pendel2],[y_car,y_pendel2],color='0.3',zorder=1,linewidth=2.0))
        
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
        M.draw(xt[i,0],xt[i,2],xt[i,4],str(round(t[i],2))+'s',image=M.image)
        #sleep(TT/float(pics))
    
    anim = animation.FuncAnimation(M.fig, animate, 
                                   frames=pics, interval=1, blit=True)
    
    
    anim.save('ex4.mp4', fps=20)
    
