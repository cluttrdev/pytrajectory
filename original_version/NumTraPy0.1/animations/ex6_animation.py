# coding=utf-8
import time
import numpy as np
from numpy import sin,cos
import pylab as plt
import matplotlib as mpl
import pickle
import math
from time import sleep

from matplotlib import animation

from ipHelp import IPS, ST, ip_syshook, dirsearch, sys  

class struct():
   def __init__(self):
      return

class Modell:
  def __init__(self):
    self.fig=plt.figure()
    self.ax=plt.axes()

    mng = plt.get_current_fig_manager()

    mng.window.wm_geometry("1000x700+50+50")  

    self.ax.set_xlim(-1.1,1.1);
    self.ax.set_ylim(-1.1,1.1);
    self.ax.set_yticks([])
    self.ax.set_xticks([])
    self.ax.set_position([0.01,0.01,0.98,0.98]);
    self.ax.set_frame_on(True);
    self.ax.set_aspect('equal')
    self.ax.set_axis_bgcolor('w');

    self.image=0

  def draw(self,phi1,phi2,frame,image=0):

    L=0.5

    x1=L*cos(phi2)
    y1=L*sin(phi2)

    x2=x1+L*cos(phi2+phi1)
    y2=y1+L*sin(phi2+phi1)

    #Init
    if (image==0):
      image=struct()

    ##update
    #else:
     #   image.sphere1.remove()
      #  image.sphere2.remove()
       # image.stab1.remove()
        #image.stab2.remove()

    #Stab
    image.stab1=self.ax.add_line(mpl.lines.Line2D([0,x1],[0,y1],color='k',zorder=0,linewidth=2.0))
    image.stab2=self.ax.add_line(mpl.lines.Line2D([x1,x2],[y1,y2],color='0.3',zorder=0,linewidth=2.0))

    image.sphere1=mpl.patches.Circle((x1,y1),0.01,color='k')
    self.ax.add_patch(image.sphere1)
    image.sphere2=mpl.patches.Circle((0,0),0.01,color='k')
    self.ax.add_patch(image.sphere2)

    #txt = plt.text(x_pendel+0.05,y_pendel,frame)

    self.image = image


    plt.draw()
    #plt.plot()




# TT=pickle.load(open('ex1_pendel_aufschwingen.pkl', 'rb'))
TT=pickle.load(open('ex6_acrobot.pkl', 'rb'))
#IPS()

A=TT['A']
xa=TT['xa']
xb=TT['xb']
l2=TT['l2']
x_sym=TT['x_sym']
u_sym=TT['u_sym']
H=TT['H']

t=A[0]
xt = A[1]
T=t[-1]-t[0]


pics = 40





ttt = np.linspace(0,(len(t)-1),pics+1,endpoint=True)

#M = Modell()
# plt.ion()
# plt.show()

# for i in ttt:
#   print i

#   M.draw(xt[i,0],xt[i,2],str(round(t[i],2))+'s',image=M.image)
#   sleep(T/float(pics))

# plt.show()



# IPS()

#M.remove()

M = Modell()

from matplotlib import animation

def animate(frame):
  i = ttt[frame]
  print frame
  M.draw(xt[i,0],xt[i,2],str(round(t[i],2))+'s',image=M.image)
  #sleep(T/float(pics))

anim = animation.FuncAnimation(M.fig, animate, 
                               frames=pics, interval=1, blit=True)

#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

anim.save('ex6.mp4', fps=20)

# plt.show()



IPS()