# coding=utf-8
import time
import numpy as np
from numpy import sin,cos
import pylab as plt
import matplotlib as mpl
import pickle
import math

from time import sleep

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
    #plt.show()




TT=pickle.load(open('ex3_aircraft.pkl', 'rb'))
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

pics = 60

M = Modell()

ttt = np.linspace(0,len(t)-1,pics+1,endpoint=True)

plt.ion()
plt.show()

for i in ttt:
  #print i
  M.draw(xt[i,0],xt[i,2],xt[i,4],str(round(t[i],2))+'s',image=M.image)
  sleep(T/float(pics))

IPS()





# from matplotlib import animation

# def animate(frame):
#   i = ttt[frame]
#   #print frame
#   M.draw(xt[i,0],xt[i,2],xt[i,4],str(round(t[i],2))+'s',image=M.image)
#   #sleep(T/float(pics))

# anim = animation.FuncAnimation(M.fig, animate, 
#                                frames=pics, interval=1, blit=True)


# anim.save('ex3.mp4', fps=20)

# # plt.show()



# IPS()


IPS()