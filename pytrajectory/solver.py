import numpy as np
from numpy.linalg import solve, norm
import scipy as scp

import log


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
    
    def __init__(self, F, DF, x0, tol=1e-2, maxIt=100, method='leven'):
        self.F = F
        self.DF = DF
        self.x0 = x0
        self.tol = tol
        self.reltol = 1e-5
        self.maxIt = maxIt
        self.method = method
        
        self.sol = None
    

    def solve(self):
        '''
        This is just a wrapper to call the chosen algorithm for solving the
        collocation equation system.
        '''
        
        if (self.method == 'leven'):
            log.info("    Run Levenberg-Marquardt method")
            self.leven()
        elif (self.method == 'new_leven'):
            self.alternate_levenberg_marquardt()
        
        
        if (self.sol == None):
            log.warn("Wrong solver, returning initial value.")
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
        res_alt = 1e10
        
        eye = scp.sparse.identity(len(self.x0))

        mu = 0.1
        
        # borders for convergence-control
        b0 = 0.2
        b1 = 0.8

        roh = 0.0

        reltol = self.reltol
        
        # New
        Fx, X, U = self.F(x)
        
        while((res > self.tol) and (self.maxIt > i) and (abs(res-res_alt) > reltol)):
            i += 1
            
            #if (i-1)%4 == 0:
            # New
            DFx = self.DF(x, X, U)
            DFx = scp.sparse.csr_matrix(DFx)
            
            while (roh < b0):                
                A = DFx.T.dot(DFx) + mu**2*eye
                b = DFx.T.dot(Fx)
                
                #s = -solve(A, b)
                s = -scp.sparse.linalg.spsolve(A,b)

                xs = x + np.array(s).flatten()
                
                # New
                Fxs, X, U = self.F(xs)

                normFx = norm(Fx)
                normFxs = norm(Fxs)

                roh = (normFx**2 - normFxs**2) / (normFx**2 - (norm(Fx+DFx.dot(s)))**2)
                
                if (roh<=b0): mu = 2.0*mu
                if (roh>=b1): mu = 0.5*mu
                #log.info("  roh= %f    mu= %f"%(roh,mu))
                
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
            log.info("      nIt= %d    res= %f"%(i,res))
            
            # NEW - experimental
            if res<1.0:
                reltol = 1e-4

        self.sol = x
    
    
    def alternate_levenberg_marquardt(self):
        '''
        This is an alternative implementation of the Levenberg-Marquardt method
        due to some bugs, probably, in the one used so far.
        '''
        
        from IPython import embed as IPS
        
        eps1 = self.tol
        eps2 = self.reltol
        
        nu = 2.0
        x = self.x0
        
        Fx = self.F(x)
        
        DFx = self.DF(x)
        DFx = scp.sparse.csr_matrix(DFx)
        
        A = DFx.T.dot(DFx)
        g = DFx.T.dot(Fx)
        
        tau = 1e-6
        mu = tau * A.max()
        
        I = scp.sparse.identity(len(x))
        
        found = (norm(g, np.inf) <= eps1)
        
        for i in xrange(self.maxIt):
            if found:
                break
            
            Q = A+mu*I
            d = scp.sparse.linalg.spsolve(Q, -g)
            
            if norm(d) <= eps2*(norm(x) + eps2):
                found = True
            else:
                x_new = x + d
                Fx_new = self.F(x_new)
                
                roh = (norm(Fx) - norm(Fx_new)) / (0.5*(d.T.dot(mu*d-g)))
                
                if roh > 0.0:
                    x = x_new
                    
                    DFx = self.DF(x)
                    DFx = scp.sparse.csr_matrix(DFx)
                    
                    A = DFx.T.dot(DFx)
                    g = DFx.T.dot(Fx_new)
                    
                    found = (norm(g, np.inf) <= eps1)
                    
                    mu = mu * max(1/3.0, 1.0 - (2.0*roh - 1.0)**3)
                    nu = 2.0
                else:
                    mu = mu * nu
                    nu = 2.0 * nu
            
            log.info("      nIt= %d    res= %f"%(i,norm(g, np.inf)))
        
        self.sol = x

