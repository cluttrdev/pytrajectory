# coding: utf8

import numpy as np
import sympy as sp
from scipy.integrate import ode
import symb_tools as st


from ipHelp import IPS, ST, ip_syshook, dirsearch, sys

class Simulation:
	#this class is used for solving the initial value problem
	def __init__(self,f,T,start,x_sym,u,u_sym,dt=0.01):


		self.dt = dt
		self.ff=f #the vector field
		self.T=T  #simulation time
		self.x_sym=x_sym #symbols of x
		self.u_sym = u_sym #symbols of u
		self.u=u 	#callable function
		self.xt=[]	#this is there the solution goes
		self.ut=[]  #same here
		self.t=[]   #for each time step


		#get the values at t=0
		self.xt.append(start)
		u=[]
		for uu in u_sym:
			u.append(self.u[uu](0.0))

		self.ut.append(u)
		self.t.append(0.0)

		#initialise our ode solver
		self.solver = ode(self.rhs)
		self.solver.set_initial_value(start)
		self.solver.set_integrator('vode', method='adams', rtol=1e-6)

	def rhs(self,t,x):

		if (0<=t<=self.T):
			u=[]
			for uu in self.u_sym:
				u.append(self.u[uu](t))
			xd=self.ff(x,u)
		else:
			u=[]
			for uu in self.u_sym:
				u.append(0.0)
			xd=self.ff(x,u)
		return xd


	def calcStep(self):
		
		x=list(self.solver.integrate(self.solver.t+self.dt))

		t=round(self.solver.t,5)
		if(0<=t<=self.T):
			self.xt.append(x)
			u=[]
			for uu in self.u_sym:
				u.append(self.u[uu](t))

			self.ut.append(u)
			self.t.append(t)

		return t,x


	def simulate(self):
		#starts simulation 
		#return an array with all data and the time steps
		out=[]
		t=0
		while(t<self.T):
			
			t,y=self.calcStep()
		return [np.array(self.t),np.array(self.xt),np.array(self.ut)]
