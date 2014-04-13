import numpy as np

import log
from log import IPS

from scipy.integrate import ode


class Simulation:
    '''
    This class does something ...
    
    
    Parameters
    ----------
    
    ff : callable
        Vectorfield of the control system
    
    T : float
        Simulation time
    
    u : callable
        Function of the input variables
    
    dt : float
        Time step
    '''
    
    def __init__(self,ff,T,start,u,dt=0.01):
        self.ff = ff
        self.T = T
        self.u = u
        self.dt = dt
        
        #this is where the solutions go
        self.xt = []
        self.ut = []
        
        # time steps
        self.t = []

        #get the values at t=0
        self.xt.append(start)
        
        u = self.u(0.0)
        
        self.ut.append(u)
        self.t.append(0.0)

        #initialise our ode solver
        self.solver = ode(self.rhs)
        self.solver.set_initial_value(start)
        self.solver.set_integrator('vode', method='adams', rtol=1e-6)
    
        
    def rhs(self,t,x):
        #if (0 <= t <= self.T):
        #    u = self.u(t)
        #    dx = self.ff(x,u)
        #else:
        #    u = self.u(t)
        #    dx = self.ff(x,u)
        u = self.u(t)
        dx= self.ff(x, u)
        return dx
    
    
    def calcStep(self):
        x = list(self.solver.integrate(self.solver.t+self.dt))
        t = round(self.solver.t,5)
        
        if(0 <= t <= self.T):
            self.xt.append(x)
            u = self.u(t)
            
            self.ut.append(u)
            self.t.append(t)
        
        return t,x
    
    def simulate(self):
        # starts simulation 
        #   return an array with all data and the time steps
        t = 0
        while(t<=self.T):
            t, y = self.calcStep()
        return [np.array(self.t),np.array(self.xt),np.array(self.ut)]
