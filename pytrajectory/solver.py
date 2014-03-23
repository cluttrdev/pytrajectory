import numpy as np
from numpy.linalg import solve, norm
import scipy as scp

import log
#from log import IPS
from IPython import embed as IPS
from time import time


class Solver:
    def __init__(self, n, F, DF, var, x0, tol=1e-2, maxx=10, algo='leven'):
        """
        # F should be a numpy array of sympy expressions
        # var should be a numpy array of all symbolic variables been used in GLS
        # x0 should be a numpy array of a guess
        # algo ... newton, gauss, leven
        # tol ... tolerance for the solver
        # maxx ... maximum number of iterations
        """

        self.x0 = x0
        self.sol = None
        self.var = var
        self.tol = tol
        self.reltol = tol #1e-3
        self.maxx = maxx
        self.algo = algo
        self.n = n
        if (not n == len(x0) and algo == 'newton'):
            log.warn("Newton needs square equation systems")
            return x0.tolist()


        self.F = F
        self.DF = DF
    

    def solve(self):

        if (self.algo == 'newton'):
            log.info( "Run Newton solver")
            log.warn(" ... not implemented")
            #self.newton()
            return
        elif (self.algo == 'gauss'):
            log.info( "Run Gauss solver")
            log.warn(" ... not implemented")
            #self.gauss()
            return
        elif (self.algo == 'leven'):
            log.info( "Run Levenberg-Marquardt method")
            self.leven()

        if (self.sol == None):
            log.warn("Wrong solver")
            return self.x0
        else:
            return self.sol


    def leven(self):
        i = 0
        x = self.x0
        res = 1
        res_alt = 1e10

        mu = 0.1

        # borders for convergence-control ##!! Ref zu Doku
        b0 = 0.2
        b1 = 0.8

        roh = 0.0

        n = len(self.x0)

        ##?? warum Bed. 1 und 3? (--> retol und abstol)
        while((res > self.tol) and (self.maxx > i) and (abs(res-res_alt) > self.reltol)):

            i += 1
            
            Fx = self.F(x)
            DFx = self.DF(x)
            
            # SPARSE
            DFx = scp.sparse.csr_matrix(DFx)
            
            while (roh < b0):
                ##?? warum J.T*F? (Gleichung (4.18) sagt: J*F)
                ## -> .T gehoert eigentlich oben hin
                
                A = DFx.T.dot(DFx) + mu**3*np.eye(n)
                b = DFx.T.dot(Fx)
                
                s = -solve(A, b)

                xs = x + np.array(s).flatten()
                
                Fxs = self.F(xs)

                normFx = norm(Fx)
                normFxs = norm(Fxs)

                roh = (normFx**2 - normFxs**2) / (normFx**2 - (norm(Fx+DFx.dot(s)))**2)
                
                #if (roh<0):
                    #print "roh<0"
                    #IPS(locals())
                    #break
                if (roh<=b0): mu = 1.5*mu
                if (roh>=b1): mu = 0.75*mu
                #log.info("  roh= %f    mu= %f"%(roh,mu))

            roh = 0.0
            x = x + np.array(s).flatten()
            res_alt = res
            res = normFx
            log.info("nIt= %d    res= %f"%(i,res))

        self.sol = x
