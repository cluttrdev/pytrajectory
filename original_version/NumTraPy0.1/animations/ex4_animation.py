# coding=utf-8
import time
import numpy as np
from numpy import sin,cos
import pylab as plt
import matplotlib as mpl
import pickle
import math

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

    # #update
    # else:
    #   image.sphere.remove()
    #   image.stab.remove()
    #   image.car.remove()

    
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
    image.stab=self.ax.add_line(mpl.lines.Line2D([x_car,x_pendel1],[y_car,y_pendel1],color='k',zorder=1,linewidth=2.0))
    image.stab=self.ax.add_line(mpl.lines.Line2D([x_car,x_pendel2],[y_car,y_pendel2],color='0.3',zorder=1,linewidth=2.0))

    #txt = plt.text(x_pendel+0.05,y_pendel,frame)

    self.image = image


    plt.draw()





TT=pickle.load(open('ex4_2xpendel_aufschwingen.pkl', 'rb'))
#TT=pickle.load(open('ex1_pendel_aufschwingen.pkl', 'rb'))
#TT=pickle.load(open('ex2_pendel.pkl', 'rb'))
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

pics = 40

M = Modell()



ttt = np.linspace(0,(len(t)-1),pics+1,endpoint=True)

for i in ttt:
  print i
  M.draw(xt[i,0],xt[i,2],xt[i,4],str(round(t[i],2))+'s',image=M.image)

plt.show()
IPS()