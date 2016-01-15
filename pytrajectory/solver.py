import numpy as np
from numpy.linalg import solve, norm
import scipy as scp

from log import logging



class Solver:
    '''
    This class provides solver for the collocation equation system.
    
    
    Parameters
    ----------
    
    F : callable
        The callable function that represents the equation system
    
    DF : callable
        The function for the jacobian matrix of the eqs
    
    x0: numpy.ndarray
        The start value for the sover
    
    tol : float
        The (absolute) tolerance of the solver
    
    maxIt : int
        The maximum number of iterations of the solver
    
    method : str
        The solver to use
    '''
    
    def __init__(self, F, DF, x0, tol=1e-5, maxIt=100, method='leven'):
        self.F = F
        self.DF = DF
        self.x0 = x0
        self.tol = tol
        self.reltol = 2e-5
        self.maxIt = maxIt
        self.method = method
        
        self.sol = None
    

    def solve(self):
        '''
        This is just a wrapper to call the chosen algorithm for solving the
        collocation equation system.
        '''
        
        if (self.method == 'leven'):
            logging.debug("Run Levenberg-Marquardt method")
            self.leven()
        
        if (self.sol is None):
            logging.warning("Wrong solver, returning initial value.")
            return self.x0
        else:
            return self.sol


    def leven(self):
        '''
        This method is an implementation of the Levenberg-Marquardt-Method
        to solve nonlinear least squares problems.
        
        For more information see: :ref:`levenberg_marquardt`
        '''
        i = 0
        x = self.x0
        res = 1
        res_alt = -1
        
        eye = scp.sparse.identity(len(self.x0))

        #mu = 1.0
        mu = 1e-4
        
        # borders for convergence-control
        b0 = 0.2
        b1 = 0.8

        roh = 0.0

        reltol = self.reltol
        
        Fx = self.F(x)
        
        while((res > self.tol) and (self.maxIt > i) and (abs(res-res_alt) > reltol)):
            i += 1
            
            #if (i-1)%4 == 0:
            DFx = self.DF(x)
            DFx = scp.sparse.csr_matrix(DFx)
            
            while (roh < b0):                
                A = DFx.T.dot(DFx) + mu**2*eye

                b = DFx.T.dot(Fx)
                    
                s = -scp.sparse.linalg.spsolve(A,b)

                xs = x + np.array(s).flatten()
                
                Fxs = self.F(xs)

                normFx = norm(Fx)
                normFxs = norm(Fxs)

                roh = (normFx**2 - normFxs**2) / (normFx**2 - (norm(Fx+DFx.dot(s)))**2)
                
                if (roh<=b0): mu = 2.0*mu
                if (roh>=b1): mu = 0.5*mu
                #logging.debug("  roh= %f    mu= %f"%(roh,mu))
                logging.debug('  mu = {}'.format(mu))
                
                # the following was believed to be some kind of bug, hence the warning
                # but that was not the case...
                #if (roh < 0.0):
                    #log.warn("Parameter roh in LM-method became negative", verb=3)
                    #from IPython import embed as IPS
                    #IPS()
            
            Fx = Fxs
            x = xs
            
            roh = 0.0
            res_alt = res
            res = normFx
            logging.debug("nIt= %d    res= %f"%(i,res))
            
            # NEW - experimental
            #if res<1.0:
            #    reltol = 1e-3

        self.sol = x
