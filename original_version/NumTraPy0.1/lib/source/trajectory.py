# coding: utf8

import os
import sys
here = os.getcwd()
#print here
sys.path.append(here) 


import numpy as np
import sympy as sp
from time import clock
import pylab as plt
#from scipy.integrate import ode
import symb_tools as st
from numpy import sqrt
from spline import CubicSpline, diff_func
from simulation import Simulation
from tools import setAxLinesBW, setFigLinesBW
import solver as Solver

#from lib.tools import clean
from ipHelp import IPS, ST, ip_syshook, dirsearch, sys

os.environ['SYMPY_USE_CACHE']='no'
if os.getenv('SYMPY_USE_CACHE') == 'no':
	print ' (cache: off)'
else: print ' (cache: on)'
 

DEBUG=False

if (DEBUG):
	from guppy import hpy
	import gc

class Trajectory:

	#ff ... Vektorfeld f
	#xa ... Zustand bei t=a a<b
	#xb .. Zustand bei t=b
	#n ... Dimension von x bzw. f
	#m ... dim u
	#algo ... welcher Lösungsansatz
	#delta ... sollen mehr Kollokationspunkte als Splineknoten verwendet werden, delta=2 (es gibt noch einen weiteren Kollokationspunkt zwischen den Splineknoten)
	#lambda_x ... Startwert für Iteration der Splineabschnitte für die Systemgrößen
	#lambda_u ... Anzahl der Splineabschnitte für den Eingang (wird nicht iteriert)
	#a,b das zeitliche Intervall 
	#v_max ... Anzahl der Iterationsschritte für lambda_x
	#kappa Faktor um die nächste Splineabschnittzahl zu bestimmen
	#tol ... Toleranz für den numerischen Solver
	#epsilon_tol ... Toleranz führ den Fehler zum Endzeitpunkt
	#find_integ ... nach Integratorketten suchen
	#gamma ... 'Randbedingungen für u'
	def __init__(self,ff,xa,xb,n,m=1,  
				algo='leven',
				delta=2,
				lambda_x=3,lambda_u=3,
				a=0.0,
				b=1.0,
				v_max=7,
				kappa=2,
				epsilon_tol=1e-2,
				tol=1e-5,
				find_integ=True,
				find_u_integ=True,
				gamma=None):

		self.ff=ff				#vector field f(x,u)

		self.algo=algo 				#which solver
		self.delta=delta			#constant for calculating collocation points
		self.lambda_x=lambda_x		#start value for number of spline parts of the system variables
		self.lambda_u=lambda_u		#number of spline parts for the manipulated variables
		self.a=a 					#left border
		self.b=b 					#right border
		self.m=m 					#dimension of the manipulated variables
		self.v_max=v_max 			#maximum number of iterations
		self.kappa=kappa			#factor the raising the spline part number lambda_x
		self.tol=tol 				#tolerance for the solver
		self.epsilon_tol=epsilon_tol 	#tolerance for the solution
		self.Diff = 2*self.epsilon_tol*np.ones(n) 
		self.find_integ=find_integ 		#looking for intergrator chains
		self.find_u_integ=find_u_integ	#looking for integrator chains for the manipulated variables
		self.gamma=gamma				#boundary values for the manipulated variable
		self.t1 = clock()				#get time 
		self.splines={}					#dictionary there the spline objects will be stored
		self.x_func={}					#dict there the callable function will be stored
		self.u_func={}					# '-'
		self.H={}						#dict for the error functions
		self.GLS=np.empty(0) 			#array for the equation system
		self.sol=np.empty(0) 			#array for the solution of the solver

		if (len(xa)==len(xb)==n):		#little check
			self.n=n 					#n is the dimension of the system
		else:
			print 'Dimension mismatch xa,xb'
			return

		#create symbolic variables for all system varaibles
		self.x_sym=([sp.symbols('x_%d' % i, type=float) for i in xrange(1,self.n+1)])

		#create symbolic variables for all manipulated variables
		if (m==1):
			self.u_sym=[sp.symbols('u')]
		else:
			self.u_sym=([sp.symbols('u_%d' % i, type=float) for i in xrange(1,self.m+1)])

		#dictionaries for boundary values
		self.xa={}
		self.xb={}
		for i,xx in enumerate(self.x_sym):
			self.xa[xx]=xa[i]
			self.xb[xx]=xb[i]


	def iteration(self):
		#here is the main loop
		#[5.1]		

		self.lambda_x_start=self.lambda_x

		#looking for intergrator chains [3.4]
		if (self.find_integ):
			self.find_integrators() 
		else: self.integ={}

		#which equation have to be solved by collocation
		#will be stored in l2
		s1=set(self.x_sym)
		s2=set(self.integ.values())
		l1=list(s1.difference(s2))
		l2=[]
		for xx in l1:
			l2.append(self.x_sym.index(xx))
		l2.sort()
		self.l2=l2

		#initiliase spoliens
		self.init_splines()	
		guess=np.empty(0)	#this will be the guess for the collocation equation system
		self.c_list=np.empty(0)


		#create first guess [5.1]
		for cc in self.c_splines:

			guess=np.hstack((guess,np.ones(len(self.c_splines[cc]))*0.1))
			
			#an array of all spline coefficients
			self.c_list = np.hstack((self.c_list,self.c_splines[cc]))


		self.guess_list=guess
		

		#create the equation system [3.1.1]
		self.equation_system()
		
		#solve it [4.3]
		self.solve()
		#write back the coefficients
		self.setCoeff()


		#this was the first iteration, now we are getting into the loop, see fig [8]

		#jetzt wird weiter iteriert
		ii=2
		while(max(abs(self.Diff))>self.epsilon_tol and ii<=self.v_max):

			#first we check if the maximum iteration number is reached or we 
			#got a satisfying solution already
			print '################################'
			print '### Next Iteration  lambda_x=', round(self.kappa*self.lambda_x), '###'
			print '################################'

			##DEBUG, print memory usage 
			if (DEBUG):
				h = hpy()
				print h.heap()

			#wa are raising the number of spline parts 
			self.lambda_x=round(self.kappa*self.lambda_x) #in jedem Schritt wird die Anzahl der Splineabschnitte für x ver#facht

			#store the old spline for getting the guess later
			old_spline = self.splines

			#restore some memory --- seems not work
			del self.splines
			del self.x_func
			del self.u_func
			del self.coeff_dic
			del self.GLS
			del self.guess_list
			del self.sol
			del self.xd_func

			#create the splines with the new number of parts
			self.init_splines()

			guess=np.empty(0)
			self.c_list=np.empty(0)

			#do that for each spline
			for cc in self.coeff:
				
				#if it is a system value
				if(self.splines[cc].type=='x'):

					#how many unknown coefficients does the ne spline have
					nn = len(self.c_splines[cc]) 

					#see [5.1]

					#and this will be the number of equations or points to evaluate the old spline
					pp = np.linspace(self.a,self.b,(nn+1),endpoint=False) 
					pp=pp[1:] 	#but we dont want to use the borders because they got the boundary values already
					#this will be our equation system
					AA = np.empty(nn,dtype=object)

					print '---> get new guess for Spline', str(cc)

					jjj=0
					#evaluate the old and new spline at all points in pp
					#they should be equal in these points
					for p in pp:
						AA[jjj]=self.splines[cc].f(p)-old_spline[cc].f(p)
						jjj+=1 

					#now we are solving this using the newton solver --- quite fast
					#equation system is linear and square but might quite big
					solver=Solver.solver(AA,self.c_splines[cc],[0]*nn,maxx=10,algo='newton')
					TT = solver.solve() # the solution
					del solver
					
					#we put this in our guess
					guess=np.hstack((guess,TT))

				else:
					#if it is a manipulated variable, just take the old solution
					guess=np.hstack((guess,self.coeff[cc]))
				#array of all coefficients 
				self.c_list=np.hstack((self.c_list,self.c_splines[cc]))	

			#the guess
			self.guess_list=guess
			#build collocation equation system 
			self.equation_system()
			#solve it
			self.solve()
			#set back coefficients 
			self.setCoeff()
			#solve the initial value problem
			self.simulate()
			#this is the solution of the simulation
			xt = self.A[1]
			self.Diff=np.empty(self.n)
			#whats the error
			print 'Difference:'
			for i,xx in enumerate(self.x_sym):
				self.Diff[i]=self.xb[xx]-xt[-1:][0][i]
				print str(xx),':', self.Diff[i]

			print max(abs(self.Diff))>self.epsilon_tol

			#increase iteration step
			ii+=1


	def find_integrators(self):

		fi=self.ff(self.x_sym,self.u_sym)

		dic={}
		for i in xrange(self.n):
			for xx in self.x_sym:
				if ((fi[i])==xx):
					dic[xx]=self.x_sym[i]
			if (self.find_u_integ):
				for uu in self.u_sym:
					if ((fi[i])==uu):
						dic[uu]=self.x_sym[i]

		self.integ = dic

	def init_splines(self):
		print '################################'
		print '### 	      Create splines    ###'
		print '################################'

		#that semms a bit confusing, this comes by solving the intergrator chains 
		#and swift the boundary values 
		#see [3.4]

		#splines will be initialised and callable function will be created 

		#there for we need to find out which spline we do have to create and which variables 
		#can be obtained by differensation 
		#finaly we need to give the boundary values the the spline created

		#quite a mess

		splines={}
		x_func={}
		u_func={}
		u_set=set(self.u_sym)

		#self.integ has the dict with the itergrator ralations
		v=self.integ.values()
		s=[]
		for vv in v:
			if(not self.integ.has_key(vv)):
				s.append(vv) 

		#now s will have all variables which a spline has to be created for


		dic={}
		#flip the dict, key becomes value 
		for ii in self.integ:
			dic[self.integ[ii]]=ii


		for xx in s:
			t=[] #hier wird die integratorkette abgelegt
			k=xx
			t.append(k)


			#now we are just working the way through the chain

			while(dic.has_key(k)):
				k=dic[k]
				t.append(k)

			if(len(u_set)!=0 and u_set.issubset([t[-1]])): 
				splines[xx]=CubicSpline(self.a,self.b,n=self.lambda_u,steady=False,tag=str(xx))
				splines[xx].type='u'
			else:
				splines[xx]=CubicSpline(self.a,self.b,n=self.lambda_x,steady=False,tag=str(xx))
				splines[xx].type='x'
			z=0
			s=''
			print '---Found Integrator----'
			for tt in t:
				s+='->'+str(tt) 
			print s
			for i,zz in enumerate(t):
				if(u_set.issubset([zz])):
					if (i==0):
						u_func[zz]=splines[xx].f
					if (i==1):
						u_func[zz]=splines[xx].d
					if (i==2):
						u_func[zz]=splines[xx].dd
				else:
					if (i==0):
						splines[xx].RB=[self.xa[zz],self.xb[zz]]
						if ((self.gamma!=None) and (splines[xx].type=='u')):
							splines[xx].RBd=self.gamma
						x_func[zz]=splines[xx].f
					if (i==1):
						splines[xx].RBd=[self.xa[zz],self.xb[zz]]
						if ((self.gamma!=None) and (splines[xx].type=='u')):
							splines[xx].RBdd=self.gamma
						x_func[zz]=splines[xx].d
					if (i==2):
						splines[xx].RBdd=[self.xa[zz],self.xb[zz]]
						x_func[zz]=splines[xx].dd

		for xx in self.x_sym:
			if(not x_func.has_key(xx)):
				splines[xx]=CubicSpline(self.a,self.b,n=self.lambda_x,RB=[self.xa[xx],self.xb[xx]],steady=True,tag=str(xx))
				splines[xx].type='x'
				x_func[xx]=splines[xx].f
		for xx in self.u_sym:

			if(not u_func.has_key(xx)):
				if (not self.gamma!=None):
					splines[xx]=CubicSpline(self.a,self.b,n=self.lambda_u,steady=False,tag=str(xx))
				else: 
					splines[xx]=CubicSpline(self.a,self.b,n=self.lambda_u,steady=False,tag=str(xx),RB=self.gamma)
				splines[xx].type='u'
				u_func[xx]=splines[xx].f


		for kk in splines:
			splines[kk].makesteady()


		xd_func={}
		for xx in self.x_sym:
			xd_func[xx]=diff_func(x_func[xx])

		c={}
		for ss in splines:
			c[ss]=splines[ss].k_sym

		#IPS()
		self.c_splines=c
		#IPS()

		self.splines= splines
		self.x_func = x_func
		self.u_func = u_func
		self.xd_func = xd_func

		print '--->Splines done'


	def equation_system(self):
		#here the collocation equation system will be build
		print '################################'
		print '###  Build equation system   ###'
		print '################################'
		#IPS()

		a=self.a
		b=self.b
		l2=self.l2	#this is a list of the index of the equation which should be solved by collocation method
		delta=self.delta
    	#calculate all collocation points [3.3]
		pp=np.linspace(a,b,(self.lambda_x*delta+1),endpoint=True)


		#set up an empty array for the equations becaus arrays are better than lists!?
		self.GLS=np.empty((self.lambda_x*delta+1)*(len(l2)),dtype=object)
		k=0 #counting variable for GLS

		#for each collocation point evaluate f(x,u)- \dot x = 0
		for p in pp:
			sys.stdout.write(".")
			sys.stdout.flush()
			#print p
			x=[]
			u=[]
			xd=[]
			for xx in self.x_sym:
				x.append(self.x_func[xx](p))
				xd.append(self.xd_func[xx](p))
			for uu in self.u_sym:
				u.append(self.u_func[uu](p))
			
			f=self.ff(x,u)
			#add equations to GLS
			for ii in l2:
				
				self.GLS[k]=(f[ii]-xd[ii])
				k+=1

		print '--->equation system done'
	
	def solve(self):

		#the guess
		guess=self.guess_list
		#the symbolic variables
		c=self.c_list
		
		#create our solver
		solver=Solver.solver(self.GLS,c,guess,maxx=20,tol=self.tol,algo=self.algo)

		#solve it
		self.sol=solver.solve()
		del solver #remove solver to get memory back? 

		#create dictionary to easily set coefficients later, setCoeff
		coeff_dic={}
		coeff_dic= dict(zip(c, self.sol)) #besser
		self.coeff_dic=coeff_dic

		#another dictionary to get association spline-splinecoeff
		coeff={}

		a=0 #used for indexing
		b=0
		#take solution and write back the coefficients for each spline
		for cc in self.c_splines:
			b+=len(self.c_splines[cc])
			coeff[cc]=self.sol[a:b]
			a=b

		self.coeff=coeff


	def setCoeff(self):
		#set uo coefficients for each spline
		print '################################'
		print '###     Set Spline Coeff     ###'
		print '################################'
		for cc in self.splines:
			self.splines[cc].setCoeff(self.coeff_dic)
	
		print '--->Done'

	def simulate(self):
		#solving the initial value problem
		#needs at least a callable function for all manupulated variables
		#see file simulation.py 
		u=self.u_func
		start=[]
		#get list as start value
		for xx in self.x_sym:
			start.append(self.xa[xx])
		T=self.b-self.a
		f=self.ff
		x_sym=self.x_sym
		u_sym = self.u_sym
		print '################################'
		print '### Start forward simulation ###'
		print '################################'
		print 'start: ', start
		S = Simulation(f,T,start,x_sym,u,u_sym)
		self.A=S.simulate()

		self.H={}
		t=self.A[0]
		#calculate the error functions H_i(t)
		for ii in self.l2:
			error=[]

			for tt in t:

				xe=[]
				xde=[]
				ue=[]
				for xx in self.x_sym:
					xe.append(self.x_func[xx](tt))
					xde.append(self.xd_func[xx](tt))
				for uu in self.u_sym:
					ue.append(self.u_func[uu](tt))
			
				f=self.ff(xe,ue)
				error.append(f[ii]-xde[ii]) 
			self.H[ii]=np.array(error,dtype=float)

		print '--->Simulation done'


	def plot(self):
		#provides graphics for each system variable, manipulated variable and error function
		#plots the solution of the simulation
		#at the end the error at the final state will be calculated

		print '################################'
		print '###            Plot          ###'
		print '################################'


		z=self.n+self.m+len(self.l2)
		z1=np.floor(sqrt(z))
		z2=np.ceil(z/z1)
		t=self.A[0]
		xt = self.A[1]
		ut= self.A[2]


		plt.rcParams['figure.subplot.bottom']=.2
		plt.rcParams['figure.subplot.top']= .95
		plt.rcParams['figure.subplot.left']=.13
		plt.rcParams['figure.subplot.right']=.95

		plt.rcParams['font.size']=16

		plt.rcParams['legend.fontsize']=16
		plt.rc('text', usetex=True)
		

		plt.rcParams['xtick.labelsize']=16
		plt.rcParams['ytick.labelsize']=16
		plt.rcParams['legend.fontsize']=20

		plt.rcParams['axes.titlesize']=26
		plt.rcParams['axes.labelsize']=26


		plt.rcParams['xtick.major.pad']='8'
		plt.rcParams['ytick.major.pad']='8'

		mm = 1./25.4 #mm to inch
		scale = 3
		fs = [100*mm*scale, 60*mm*scale]

		fff=plt.figure(figsize=fs, dpi=80)


		PP=1
		#IPS()
		for i,xx in enumerate(self.x_sym):
			plt.subplot(int(z1),int(z2),PP)
			PP+=1
			lines = plt.plot(t,xt[:,i])     
			plt.xlabel(r'$t$')
			#plt.ylabel('$'+str(xx)+'$')
			plt.title(r'$'+str(xx)+'(t)$')  

		#IPS()
		for i,uu in enumerate(self.u_sym):
			plt.subplot(int(z1),int(z2),PP)
			PP+=1
			lines = plt.plot(t,ut[:,i])     
			plt.xlabel(r'$t$')
			#plt.ylabel('$'+str(uu)+'(t)$')
			plt.title(r'$'+str(uu)+'(t)$')  

		for hh in self.H:

			#IPS()
			plt.subplot(int(z1),int(z2),PP)
			PP+=1
			lines = plt.plot(t,self.H[hh])     
			plt.xlabel(r'$t$')
			#plt.ylabel('$H_'+str(ii+1)+'(t)$')
			plt.title(r'$H_'+str(hh+1)+'(t)$')  

		setFigLinesBW(fff)

		plt.tight_layout()

		print 'Ending up with:'
		for i,xx in enumerate(self.x_sym):
			print str(xx),':', xt[-1:][0][i]

		print 'Should be:'
		for i,xx in enumerate(self.x_sym):
			print str(xx),':', self.xb[xx]

		print 'Difference:'
		for i,xx in enumerate(self.x_sym):
			print str(xx),':', self.xb[xx]-xt[-1:][0][i]

		plt.show()




if __name__ == '__main__':

	from sympy import cos, sin
	def f(x,u):
		x1,x2 = x
		u=u[0]

		l=0.5
		g=9.81
		ff = [	2*x2,
				2*u]
		return ff
			


	xa=[0,0]
	xb=[1,0]
	T=Trajectory(f,xa,xb,n=2,m=1)
	T.gamma=[0,0]
	T.iteration()
	IPS()






		# ### wie sehen die splines am anfang aus
		# self.init_splines()	# die ersten Splines bauen
		# self.sol=guess
		# self.setCoeff()
		# pp = np.linspace(self.a,self.b,100)
		
		# plt.rcParams['figure.subplot.bottom']=.2
		# plt.rcParams['figure.subplot.top']= .95
		# plt.rcParams['figure.subplot.left']=.13
		# plt.rcParams['figure.subplot.right']=.95

		# plt.rcParams['font.size']=16

		# plt.rcParams['legend.fontsize']=16
		# plt.rc('text', usetex=True)
		        

		# plt.rcParams['xtick.labelsize']=16
		# plt.rcParams['ytick.labelsize']=16
		# plt.rcParams['legend.fontsize']=20

		# plt.rcParams['axes.titlesize']=26
		# plt.rcParams['axes.labelsize']=26


		# plt.rcParams['xtick.major.pad']='8'
		# plt.rcParams['ytick.major.pad']='8'

		# mm = 1./25.4 #mm to inch
		# scale = 3
		# fs = [100*mm*scale, 30*mm*scale]

		# fff=plt.figure(figsize=fs, dpi=80)
		# #IPS()
		# aa=[]
		# for p in pp:
		# 	aa.append(self.splines[self.x_sym[0]].f(p))
						
		# plt.subplot(1,3,1)
		# lines = plt.plot(pp,aa)    
		# plt.xlabel(r'$t/s$')
		# plt.title(r'$x_1(t)$')  


		# aa=[]
		# for p in pp:
		# 	aa.append(self.splines[self.x_sym[1]].f(p))
						
		# plt.subplot(1,3,2)
		# lines = plt.plot(pp,aa)
		# plt.xlabel(r'$t/s$')
		# plt.title(r'$x_2(t)$')      	

		# aa=[]
		# for p in pp:
		# 	aa.append(self.splines[self.u_sym[0]].f(p))
						
		# plt.subplot(1,3,3)
		# lines = plt.plot(pp,aa)
		# plt.xlabel(r'$t/s$')
		# plt.title(r'$u(t)$')     
		
		# setFigLinesBW(fff)
		# plt.tight_layout()
		# plt.show()






