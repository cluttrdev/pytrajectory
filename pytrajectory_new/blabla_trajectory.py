
import numpy as np
import sympy as sp
import scipy as scp
from scipy import sparse
import pickle

from spline import CubicSpline, fdiff
from solver import Solver
from simulation import Simulation
from utilities import IntegChain, plotsim
import log

# DEBUGGING
DEBUG = True


class Trajectory():
    def plot(self):
        '''
        Plot the calculated trajectories and error functions.

        This method calculates the error functions and then calls
        the :func:`utilities.plotsim` function.
        '''

        try:
            import matplotlib
        except ImportError:
            log.error('Matplotlib is not available for plotting.')
            return

        # calculate the error functions H_i(t)
        max_con_err, error = consistency_error((self.a,self.b), self.x, self.u, 
                                                self.dx, self.ff, len(self.sim[0]), True)
        H = dict()
        for i in self.eqind:
            H[i] = error[:,i]

        # call utilities.plotsim()
        #plotsim(self.sim, H)
        t = self.sim[0]
        xt = np.array([self.x(tt) for tt in t])
        ut = self.sim[2]
        plotsim([t,xt,ut], H)


    def save(self, fname=None):
        '''
        Save system data, callable solution functions and simulation results.
        '''

        save = dict()

        # system data
        #save['ff_sym'] = self.ff_sym
        #save['ff'] = self.ff
        #save['a'] = self.a
        #save['b'] = self.b

        # boundary values
        #save['xa'] = self.xa
        #save['xb'] = self.xb
        #save['uab'] = self.uab

        # solution functions       
        #save['x'] = self.x
        #save['u'] = self.u
        #save['dx'] = self.dx

        # simulation results
        save['sim'] = self.sim
        
        if not fname:
            fname = __file__.split('.')[0] + '.pkl'
        elif not fname.endswith('.pkl'):
            fname += '.pkl'
        
        with open(fname, 'wb') as dumpfile:
            pickle.dump(save, dumpfile)


    def clear(self):
        '''
        This method is intended to delete some attributes of the object that
        are no longer neccessary after the iteration has finished.

        TODO: extend (may be not necessary anymore...)
        '''
        
        del self.c_list
        
        try:
            self.old_splines
        except:
            pass



