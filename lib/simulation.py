import numpy as np

import log
from log import IPS

from scipy.integrate import ode


class Simulation:
    def __init__(self,ff,T,start,x_sym,u_sym,u,dt=0.01):
        self.ff = ff         # ff    ... vectorfield
        self.T = T           # T     ... simulation time
        self.x_sym = x_sym   # x_sym ... symbolic system variables
        self.u_sym = u_sym   # u_sym ... symbolic manipulated variables
        self.u = u           # u     ... callable function
        self.dt = dt         # dt    ... time step
        
        #this is where the solutions go
        self.xt = []
        self.ut = []
        
        # time steps
        self.t = []

        #get the values at t=0
        self.xt.append(start)
        u = []
        for uu in u_sym:
            u.append(self.u[uu](0.0))
        
        self.ut.append(u)
        self.t.append(0.0)

        #initialise our ode solver
        self.solver = ode(self.rhs)
        self.solver.set_initial_value(start)
        self.solver.set_integrator('vode', method='adams', rtol=1e-6)
    
        
    def rhs(self,t,x):
        if (0 <= t <= self.T):
            u = []
            for uu in self.u_sym:
                u.append(self.u[uu](t))
            dx = self.ff(x,u)
        else:
            u = []
            for uu in self.u_sym:
                u.append(0.0)
            dx = self.ff(x,u)
        return dx
    
    
    def calcStep(self):
        x = list(self.solver.integrate(self.solver.t+self.dt))
        t = round(self.solver.t,5)
        
        if(0 <= t <= self.T):
            self.xt.append(x)
            u = []
            for uu in self.u_sym:
                u.append(self.u[uu](t))

            self.ut.append(u)
            self.t.append(t)
        
        return t,x
    
    def simulate(self):
        # starts simulation 
        #   return an array with all data and the time steps
        t = 0
        while(t<self.T):
            t, y = self.calcStep()
        return [np.array(self.t),np.array(self.xt),np.array(self.ut)]
