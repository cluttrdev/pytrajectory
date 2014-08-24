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
    
    maxx : int
        The maximum number of iterations of the solver
    
    algo : str
        The solver to use
    '''
    
    def __init__(self, F, DF, x0, tol=1e-2, maxx=10, algo='leven'):
        self.F = F
        self.DF = DF
        self.x0 = x0
        self.tol = tol
        self.reltol = tol
        #self.reltol = 1e-2
        self.maxx = maxx
        self.algo = algo
        
        self.sol = None
    

    def solve(self):
        '''
        This is just a wrapper to call the chosen algorithm for solving the
        collocation equation system.
        '''
        
        if (self.algo == 'newton'):
            #log.info( "Run Newton solver")
            #self.newton()
            log.warn('Not yet implemented. Please use "leven"-algorithm!')
            self.leven()
        elif (self.algo == 'gauss'):
            #log.info( "Run Gauss solver")
            #self.gauss()
            log.warn('Not yet implemented. Please use "leven"-algorithm!')
            self.leven()
        elif (self.algo == 'leven'):
            log.info("    Run Levenberg-Marquardt method")
            self.leven()
            
        
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
        
        eye = np.eye(len(self.x0))

        mu = 0.1

        # borders for convergence-control
        b0 = 0.2
        b1 = 0.8

        roh = 0.0

        reltol = self.reltol
        while((res > self.tol) and (self.maxx > i) and (abs(res-res_alt) > reltol)):
            i += 1
            
            Fx = self.F(x)
            DFx = self.DF(x)
            
            # NEW -experimental
            if res >= 1:
                DFx = self.DF(x)
            
            # SPARSE
            DFx = scp.sparse.csr_matrix(DFx)
            
            while (roh < b0):                
                A = DFx.T.dot(DFx) + mu**2*eye
                b = DFx.T.dot(Fx)
                
                s = -solve(A, b)

                xs = x + np.array(s).flatten()
                
                Fxs = self.F(xs)

                normFx = norm(Fx)
                normFxs = norm(Fxs)

                roh = (normFx**2 - normFxs**2) / (normFx**2 - (norm(Fx+DFx.dot(s)))**2)
                
                if (roh<=b0): mu = 2.0*mu
                if (roh>=b1): mu = 0.5*mu
                #log.info("  roh= %f    mu= %f"%(roh,mu))

            roh = 0.0
            x = x + np.array(s).flatten()
            res_alt = res
            res = normFx
            log.info("      nIt= %d    res= %f"%(i,res))
            
            # NEW - experimental
            #if res<1.0:
            #    reltol = 1e-3

        self.sol = x
    
    
    def gauss(self):
        i = 0
        x = self.x0
        res = 1
        res_alt = 10e10
        while((res>self.tol) and (self.maxx>i) and (abs(res-res_alt)>self.tol)):
            i += 1
            r = self.F(x)

            D = self.DF(x)
            DD = np.linalg.solve(D.T*D,D.T*r.T) 

            x = np.matrix(x).T - DD
            x = np.array(x.flatten())[0]
            res_alt = res
            res = norm(r)
            print i,': ',res

        self.sol = x
    
    
    def newton(self):
        res = 1
        i = 0
        x = self.x0
        Fx = self.F(x)
        
        while(res>self.tol and self.maxx>i):
            i += 1
            DFx = self.DF(x)

            h=np.array(np.linalg.solve(DFx,Fx.T).flatten())[0] 

            x -= h

            Fx = np.matrix(self.F(x))
            res = np.linalg.norm(Fx)
            print i,': ',res
        
        self.sol = x
