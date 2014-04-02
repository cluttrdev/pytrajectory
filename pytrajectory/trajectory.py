import os
import sys

import numpy as np
import sympy as sp
from scipy import sparse

import matplotlib.pyplot as plt

from spline import CubicSpline, fdiff
from solver import Solver
from simulation import Simulation
from tools import IntegChain

import log
#from log import IPS
from time import time

from IPython import embed as IPS


class Trajectory():
    '''
    Base class of the PyTrajectory project.


    :param callable ff: Vectorfield (rhs) of the control system
    :param real a: Left border
    :param real b: Right border
    :param list xa: Boundary values at the left border
    :param list xb: Boundary values at the right border
    :param list g: Boundary values of the input variables
    :param int sx: Initial number of spline parts for the system variables
    :param int su: Initial number of spline parts for the input variables
    :param int kx: Factor for raising the number of spline parts for the system variables
    :param int delta: Constant for calculation of collocation points
    :param int maxIt: Maximum number of iterations
    :param real eps: Tolerance for the solution of the initial value problem
    :param real tol: Tolerance for the solver of the equation system
    :param str algo: Solver to use
    :param bool use_chains: Whether or not to use integrator chains
    '''

    def __init__(self, ff, a=0.0, b=1.0, xa=None, xb=None, g=None, sx=5, su=5, 
                 kx=5, delta=2, maxIt=7, eps=1e-2, tol=1e-5, algo='leven', 
                 use_chains=True):

        # save symbolic vectorfield
        self.ff_sym = ff

        self.algo = algo
        self.delta = delta
        self.sx = sx
        self.su = su
        self.a = a
        self.b = b
        self.maxIt = maxIt
        self.kx = kx
        self.eps = eps
        self.tol = tol
        self.use_chains = use_chains
        self.g = g

        # system analysis
        sys_ana = analyseSystem(ff)

        self.n = sys_ana['n']
        self.m = sys_ana['m']
        self.chains = sys_ana['chains']
        #self.su = sys_ana['su'] --> not implemented yet

        # a little check
        if not (len(xa) == len(xb) == self.n):
            raise ValueError, 'Dimension mismatch xa,xb'

        # iteration number
        self.nIt = 0

        # error of simulation
        self.err = 2*self.eps*np.ones(self.n)

        # dictionary for spline objects
        #   key: variable   value: CubicSpline-object
        self.splines = dict()

        # dictionaries for callable solution functions
        self.x_fnc = dict()
        self.u_fnc = dict()

        # create symbolic variables
        self.x_sym = ([sp.symbols('x%d' % i, type=float) for i in xrange(1, self.n+1)])
        self.u_sym = ([sp.symbols('u%d' % i, type=float) for i in xrange(1, self.m+1)])

        # transform symbolic ff to numeric ff for faster evaluation
        F = sp.Matrix(ff(self.x_sym,self.u_sym))
        _ff_num = sp.lambdify(self.x_sym+self.u_sym, F, modules='numpy')
        
        def ff_num(x, u):
            xu = np.hstack((x, u))
            return _ff_num(*xu)
        
        self.ff = ff_num

        # dictionaries for boundary conditions
        self.xa = dict()
        self.xb = dict()
        for i, xx in enumerate(self.x_sym):
            self.xa[xx] = xa[i]
            self.xb[xx] = xb[i]

        # set logfile
        #log.set_file()


    def startIteration(self):
        '''
        This is the main loop --> [5.1]
        '''

        log.info( 40*"#")
        log.info("       ---- First Iteration ----")
        log.info( 40*"#")
        log.info("# spline parts: %d"%self.sx)

        # looking for integrator chains --> [3.4]
        if not self.use_chains:
            self.chains = dict()

        # determine which equations have to be solved by collocation
        eqind = []

        if self.chains:
            for ic in self.chains:
                if ic.lower.name.startswith('x'):
                    lower = ic.lower
                    eqind.append(self.x_sym.index(lower))
                    eqind.sort()
        else:
            eqind = range(self.n)

        self.eqind = np.array(eqind)

        # start first iteration
        self.iterate()

        ######################
        # this is the solution of the simulation
        xt = self.A[1]
        self.err = np.empty(self.n)

        # what is the error
        log.info("Difference:")
        for i, xx in enumerate(self.x_sym):
            self.err[i] = self.xb[xx] - xt[-1:][0][i]
            log.info(str(xx)+" : %f"%self.err[i])
        
        errmax = max(abs(self.err))
        log.info("--> reached desired accuracy: "+str(errmax <= self.eps))
        ######################

        # this was the first iteration
        # now we are getting into the loop, see fig [8]
        while(max(abs(self.err)) > self.eps and self.nIt < self.maxIt):
            log.info( 40*"#")
            log.info("       ---- Next Iteration ----")
            log.info( 40*"#")

            # raise the number of spline parts
            self.sx = round(self.kx*self.sx)

            log.info("# spline parts: %d"%self.sx)

            # store the old spline for getting the guess later
            self.old_splines = self.splines

            # start next iteration
            self.iterate()

            # this is the solution of the simulation
            xt = self.A[1]
            self.err = np.empty(self.n)

            # what is the error
            log.info("Difference:")
            for i, xx in enumerate(self.x_sym):
                self.err[i] = abs(self.xb[xx] - xt[-1:][0][i])
                log.info(str(xx)+" : %f"%self.err[i])
            
            errmax = max(abs(self.err))
            log.info("--> reached desired accuracy: "+str(errmax <= self.eps))
        
        log.info(40*"#")
    
    
    def iterate(self):
        '''
        This method is used to run one iteration
        '''
        self.nIt += 1

        # initialise splines
        with log.Timer("initSplines()"):
            self.initSplines()

        # Get first guess for solver
        with log.Timer("getGuess()"):
            self.getGuess()

        # create equation system --> [3.1.1]
        with log.Timer("buildEQS()"):
            self.buildEQS()

        # solve it --> [4.3]
        with log.Timer("solve()"):
            self.solve()

        # write back the coefficients
        with log.Timer("setCoeff()"):
            self.setCoeff()

        # solve the initial value problem
        with log.Timer("simulate()"):
            self.simulate()


    def getGuess(self):
        '''
        This method is used to determine a starting value (guess) for the
        solver of the collocation equation system --> docu p. 24
        '''

        if (self.nIt == 1):
            self.c_list = np.empty(0)

            for k, v in sorted(self.indep_coeffs.items(), key = lambda (k, v): k.name):
                self.c_list = np.hstack((self.c_list, v))
            guess = 0.1*np.ones(len(self.c_list))
        else:
            # make splines local
            old_splines = self.old_splines
            new_splines = self.splines

            guess = np.empty(0)
            self.c_list = np.empty(0)

            # get new guess for every independent variable
            for k, v in sorted(self.coeffs_sol.items(), key = lambda (k, v): k.name):
                self.c_list = np.hstack((self.c_list, self.indep_coeffs[k]))

                if (new_splines[k].type == 'x'):
                    log.info("get new guess for spline %s"%k.name)

                    # how many unknown coefficients does the new spline have
                    nn = len(self.indep_coeffs[k])

                    # and this will be the points to evaluate the old spline in
                    #   but we don't want to use the borders because they got
                    #   the boundary values already
                    gpts = np.linspace(self.a, self.b, (nn+1))[1:]

                    # evaluate the old and new spline at all points in gpts
                    #   they should be equal in these points

                    OLD = [None]*len(gpts)
                    NEW = [None]*len(gpts)
                    NEW_abs = [None]*len(gpts)
                    for i, p in enumerate(gpts):
                        OLD[i] = old_splines[k].f(p)
                        NEW[i], NEW_abs[i] = new_splines[k].tmp_f(p)

                    OLD = np.array(OLD)
                    NEW = np.array(NEW)
                    NEW_abs = np.array(NEW_abs)
                    
                    try:
                        TT = np.linalg.solve(NEW, OLD-NEW_abs)
                    except Exception as err:
                        if not isinstance(err, np.linalg.linalg.LinAlgError):
                            print type(err)
                            raise err
                        else:
                            log.warn("numpy encountered singular matrix")
                            #F = lambda c: np.dot(NEW, c) + NEW_abs - OLD
                            #DF = lambda c: NEW
                            #x0 = np.zeros(OLD.shape)
                            #S = Solver(F=F, DF=DF, x0=x0, algo='newton')
                            #TT = S.solve()
                            #IPS()
                            
                            # --> try to fix this
                            NEW = NEW[:-1,:-1]
                            TT = np.linalg.solve(NEW, (OLD-NEW_abs)[:-1])
                            TT = np.hstack((TT, TT.mean()))

                    guess = np.hstack((guess, TT))
                else:
                    #if it is a manipulated variable, just take the old solution
                    guess = np.hstack((guess, self.coeffs_sol[k]))

        # the new guess
        self.guess = guess


    def initSplines(self):
        '''
        This method is used to initialise the temporary splines
        '''
        log.info( 40*"#")
        log.info( "#########  Initialise Splines  #########")
        log.info( 40*"#")

        # dictionaries for splines and callable solution function for x,u and dx
        splines = dict()
        x_fnc = dict()
        u_fnc = dict()
        dx_fnc = dict()

        chains = self.chains

        # first handle variables that are part of an integrator chain
        for ic in chains:
            upper = ic.upper

            # here we just create a spline object for the upper ends of every chain
            # w.r.t. its lower end
            if ic.lower.name.startswith('x'):
                splines[upper] = CubicSpline(self.a,self.b,n=self.sx,
                                            bc=[self.xa[upper],self.xb[upper]],
                                            steady=False,tag=upper.name)
                splines[upper].type = 'x'
            elif ic.lower.name.startswith('u'):
                splines[upper] = CubicSpline(self.a,self.b,n=self.su,
                                            bc=self.g,steady=False,
                                            tag=upper.name)
                splines[upper].type = 'u'

            for i,elem in enumerate(ic.elements):
                if elem in self.u_sym:
                    if (i == 0):
                        u_fnc[elem] = splines[upper].tmp_f
                    if (i == 1):
                        u_fnc[elem] = splines[upper].tmp_df
                    if (i == 2):
                        u_fnc[elem] = splines[upper].tmp_ddf
                elif elem in self.x_sym:
                    if (i == 0):
                        splines[upper].bc = [self.xa[elem],self.xb[elem]]
                        if ((self.g != None) and (splines[upper].type == 'u')):
                            splines[upper].bcd = self.g
                        x_fnc[elem] = splines[upper].tmp_f
                    if (i == 1):
                        splines[upper].bcd = [self.xa[elem],self.xb[elem]]
                        if ((self.g != None) and (splines[upper].type == 'u')):
                            splines[upper].bcdd = self.g
                        x_fnc[elem] = splines[upper].tmp_df
                    if (i == 2):
                        splines[upper].bcdd = [self.xa[elem],self.xb[elem]]
                        x_fnc[elem] = splines[upper].tmp_ddf

        # now handle the variables which are not part of any chain
        for xx in self.x_sym:
            if (not x_fnc.has_key(xx)):
                splines[xx] = CubicSpline(self.a,self.b,n=self.sx,
                                        bc=[self.xa[xx],self.xb[xx]],
                                        steady=False,tag=str(xx))
                splines[xx].type = 'x'
                x_fnc[xx] = splines[xx].tmp_f

        for uu in self.u_sym:
            if (not u_fnc.has_key(uu)):
                splines[uu] = CubicSpline(self.a,self.b,n=self.su,
                                        bc=self.g,steady=False,tag=str(uu))
                splines[uu].type = 'u'
                u_fnc[uu] = splines[uu].tmp_f

        # solve smoothness conditions of each spline
        for ss in splines:
            with log.Timer("makesteady()"):
                splines[ss].makesteady()

        for xx in self.x_sym:
            dx_fnc[xx] = fdiff(x_fnc[xx])

        indep_coeffs = dict()
        for ss in splines:
            indep_coeffs[ss] = splines[ss].c_indep

        self.indep_coeffs = indep_coeffs

        self.splines = splines
        self.x_fnc = x_fnc
        self.u_fnc = u_fnc
        self.dx_fnc = dx_fnc


    def buildEQS(self):
        '''
        This method is used to build the collocation equation system 
        that will be solved later.
        '''

        log.info( 40*"#")
        log.info("####  Building the equation system  ####")
        log.info( 40*"#")

        # make functions local
        x_fnc = self.x_fnc
        dx_fnc = self.dx_fnc
        u_fnc = self.u_fnc

        # make symbols local
        x_sym = self.x_sym
        u_sym = self.u_sym

        a = self.a
        b = self.b
        delta = self.delta

        # generate collocation points ---> [3.3]
        cpts = np.linspace(a, b, (self.sx*delta+1), endpoint=True)
        self.cpts = cpts

        lx = len(cpts)*len(x_sym)
        lu = len(cpts)*len(u_sym)

        Mx = [None]*lx
        Mx_abs = [None]*lx
        Mdx = [None]*lx
        Mdx_abs = [None]*lx
        Mu = [None]*lu
        Mu_abs = [None]*lu

        indic = dict()
        i = 0
        j = 0
        # iterate over spline quantities
        for k, v in sorted(self.indep_coeffs.items(), key=lambda (k, v): k.name):
            # increase j by the number of indep coeffs on which it depends
            j += len(v)
            indic[k] = (i, j)
            i = j

        # iterate over all quantities including inputs
        for sq in x_sym+u_sym:
            for ic in self.chains:
                if sq in ic:
                    indic[sq] = indic[ic.upper]

        # total number of indep coeffs
        c_len = len(self.c_list)

        eqx = 0
        equ = 0
        for i, p in enumerate(cpts):
            for xx in x_sym:
                mx = np.zeros(c_len)
                mdx = np.zeros(c_len)

                i, j = indic[xx]

                mx[i:j], Mx_abs[eqx] = x_fnc[xx](p)
                mdx[i:j], Mdx_abs[eqx] = dx_fnc[xx](p)

                Mx[eqx] = mx
                Mdx[eqx] = mdx
                eqx += 1

            for uu in u_sym:
                mu = np.zeros(c_len)

                i, j = indic[uu]

                mu[i:j], Mu_abs[equ] = u_fnc[uu](p)

                Mu[equ] = mu
                equ += 1

        self.Mx = np.array(Mx)
        self.Mx_abs = np.array(Mx_abs)
        self.Mdx = np.array(Mdx)
        self.Mdx_abs = np.array(Mdx_abs)
        self.Mu = np.array(Mu)
        self.Mu_abs = np.array(Mu_abs)

        ################################################################
        # SPARSE
        #self.Mx = sparse.csr_matrix(self.Mx)
        #self.Mx_abs = sparse.csr_matrix(self.Mx_abs)
        #self.Mdx = sparse.csr_matrix(self.Mdx)
        #self.Mdx_abs = sparse.csr_matrix(self.Mdx_abs)
        #self.Mu = sparse.csr_matrix(self.Mu)
        #self.Mu_abs = sparse.csr_matrix(self.Mu_abs)

        ################################################################
        # for creation of the jacobian matrix

        f = self.ff_sym(x_sym, u_sym)
        Df = sp.Matrix(f).jacobian(self.x_sym+self.u_sym)

        self.Df = sp.lambdify(self.x_sym+self.u_sym, Df, modules='numpy')

        ################################################################


    def solve(self):
        '''
        This method is used to solve the collocation equation system.
        '''

        log.info( 40*"#")
        log.info("#####  Solving the equation system  ####")
        log.info( 40*"#")

        # create our solver
        solver = Solver(self.G, self.DG, self.guess, maxx= 20, tol= self.tol,
                        algo=self.algo)

        # solve the equation system
        self.sol = solver.solve()


    def G(self, c):
        '''
        This is the callable function that represents the collocation system.
        '''

        ff = self.ff
        eqind = self.eqind

        x_len = len(self.x_sym)
        u_len = len(self.u_sym)

        # DENSE
        X = np.dot(self.Mx, c) + self.Mx_abs
        U = np.dot(self.Mu, c) + self.Mu_abs

        # SPARSE
        #X = Mx.dot(c) + Mx_abs
        #U = Mu.dot(c) + Mu_abs

        X = np.array(X).reshape((-1, x_len))
        U = np.array(U).reshape((-1, u_len))

        # evaluate system equations and select those related
        # to lower ends of integrator chains (via eqind)
        # other equations need not to be solved
        F = np.array([ff(x, u) for x, u in zip(X, U)], dtype=float).squeeze()[:,eqind]

        # DENSE
        dX = np.dot(self.Mdx,c) + self.Mdx_abs
        dX = np.reshape(dX,(-1,x_len))[:,eqind]

        # SPARSE
        #dX = Mdx.dot(c) + Mdx_abs
        #dX = np.array(dX).reshape((-1,x_len))[:,eqind]

        G = F-dX

        return G.flatten()


    def DG(self, c):
        '''
        This is the callable function that returns the jacobian matrix 
        of the collocation system.
        '''

        Df = self.Df
        eqind = self.eqind

        x_len = len(self.x_sym)
        u_len = len(self.u_sym)

        # x-/u-values in all collocation points
        X = self.Mx.dot(c) + self.Mx_abs
        X = np.array(X).reshape((-1,x_len)) # one column for every state component
        U = self.Mu.dot(c) + self.Mu_abs
        U = np.array(U).reshape((-1,u_len)) # one column for every input component

        # construct jacobian of rhs w.r.t. indep coeffs
        DF_blocks = []
        for x,u in zip(X,U):
            # get one row of U and X respectively
            tmp_xu = np.hstack((x,u))

            # evaluate jacobian at current collocation point
            DF_blocks.append(Df(*tmp_xu))

        #tmp= np.array(tmp)
        #assert tmp.shape== (len(self.cpts), x_len, len(self.c_list))

        # SPARSE
        #Mu = Mu.toarray()
        #Mx = Mx.toarray()

        DF = []
        for i,df in enumerate(DF_blocks):
            J_XU = np.vstack(( self.Mx[x_len*i:x_len*(i+1)], self.Mu[u_len*i:u_len*(i+1)] ))
            res = np.dot(df,J_XU)
            #assert res.shape == (x_len,len(self.c_list))
            DF.append(res)

        DF = np.array(DF)[:,eqind,:]
        # 1st index : collocation point
        # 2nd index : equations that have to be solved --> end of an integrator chain
        # 3rd index : component of c

        # now compute jacobian of x_dot w.r.t. to indep coeffs
        # DENSE
        DdX = self.Mdx.reshape((len(self.cpts),-1,len(self.c_list)))[:,eqind,:]

        # SPARSE
        #DdX = Mdx.toarray().reshape((len(self.cpts),-1,len(self.c_list)))[:,eqind,:]

        # stack matrices in vertical direction
        DG = np.vstack(DF) - np.vstack(DdX)

        return DG


    def setCoeff(self):
        '''
        This method is used to create the actual splines by using the numerical 
        solutions to set up the coefficients of the polynomial spline parts of 
        every created spline.
        '''

        log.info("Set spline coefficients")

        sol = self.sol
        subs = dict()

        for k, v in sorted(self.indep_coeffs.items(), key=lambda (k, v): k.name):
            i = len(v)
            subs[k] = sol[:i]
            sol = sol[i:]

        for var in self.x_sym + self.u_sym:
            for ic in self.chains:
                if var in ic:
                    subs[var] = subs[ic.upper]

        # set numerical coefficients for each spline
        for cc in self.splines:
            self.splines[cc].set_coeffs(subs[cc])

        # reset callable functions
        self.x_fnc = dict()
        self.u_fnc = dict()
        self.dx_fnc = dict()

        # again we handle variables that are part of an integrator chain first
        for ic in self.chains:
            upper = ic.upper

            for i,elem in enumerate(ic.elements):
                if elem in self.u_sym:
                    if (i == 0):
                        self.u_fnc[elem] = self.splines[upper].f
                    if (i == 1):
                        self.u_fnc[elem] = self.splines[upper].df
                    if (i == 2):
                        self.u_fnc[elem] = self.splines[upper].ddf
                elif elem in self.x_sym:
                    if (i == 0):
                        self.x_fnc[elem] = self.splines[upper].f
                    if (i == 1):
                        self.x_fnc[elem] = self.splines[upper].df
                    if (i == 2):
                        self.x_fnc[elem] = self.splines[upper].ddf

        # now handle the variables which are not part of any chain
        for xx in self.x_sym:
            if (not self.x_fnc.has_key(xx)):
                self.x_fnc[xx] = self.splines[xx].f

        for uu in self.u_sym:
            if (not self.u_fnc.has_key(uu)):
                self.u_fnc[uu] = self.splines[uu].f

        for xx in self.x_sym:
            self.dx_fnc[xx] = fdiff(self.x_fnc[xx])

        # yet another dictionary for solution and coeffs
        coeffs_sol = dict()

        # used for indexing
        i = 0
        j = 0

        # take solution and write back the coefficients for each spline
        for k, v in sorted(self.indep_coeffs.items(), key=lambda (k, v): k.name):
            j += len(v)
            coeffs_sol[k] = self.sol[i:j]
            i = j

        self.coeffs_sol = coeffs_sol


    def simulate(self):
        '''
        This method is used to solve the initial value problem.
        '''

        log.info( 40*"#")
        log.info("##  Solving the initial value problem ##")
        log.info( 40*"#")

        # get list as start value
        start = []
        for xx in self.x_sym:
            start.append(self.xa[xx])
        log.info("start: %s"%str(start))

        T = self.b - self.a

        S = Simulation(self.ff, T, start, self.u)

        self.A = S.simulate()

        self.H = dict()

        t = self.A[0]

        #calculate the error functions H_i(t)
        for ii in self.eqind:
            error = []

            for tt in t:
                xe = []
                xde = []
                ue = []
                for xx in self.x_sym:
                    xe.append(self.x_fnc[xx](tt))
                    xde.append(self.dx_fnc[xx](tt))
                for uu in self.u_sym:
                    ue.append(self.u_fnc[uu](tt))

                f = self.ff_sym(xe,ue)
                error.append(f[ii]-xde[ii])
            self.H[ii] = np.array(error,dtype=float)

        log.info(40*"-")


    def x(self, t):
        '''
        This function returns the system state at a given (time-) point.
        '''
        return np.array([self.x_fnc[xx](t) for xx in self.x_sym])


    def u(self, t):
        '''
        This function returns the inputs state at a given (time-) point.
        '''
        return np.array([self.u_fnc[uu](t) for uu in self.u_sym])


    def plot(self):
        '''
        This method provides graphics for each system variable, manipulated
        variable and error function and plots the solution of the simulation.
        '''

        log.info("Plot")

        z=self.n+self.m+len(self.eqind)
        z1=np.floor(np.sqrt(z))
        z2=np.ceil(z/z1)
        t=self.A[0]
        xt = self.A[1]
        ut= self.A[2]


        log.info("Ending up with:")
        for i,xx in enumerate(self.x_sym):
            log.info(str(xx)+" : "+str(xt[-1:][0][i]))

        log.info("Shoul be:")
        for i,xx in enumerate(self.x_sym):
            log.info(str(xx)+" : "+str(self.xb[xx]))

        log.info("Difference")
        for i,xx in enumerate(self.x_sym):
            log.info(str(xx)+" : "+str(self.xb[xx]-xt[-1:][0][i]))


        def setAxLinesBW(ax):
            """
            Take each Line2D in the axes, ax, and convert the line style to be
            suitable for black and white viewing.
            """
            MARKERSIZE = 3


            ##?? was bedeuten die Zahlen bei dash[...]?
            COLORMAP = {
                'b': {'marker': None, 'dash': (None,None)},
                'g': {'marker': None, 'dash': [5,5]},
                'r': {'marker': None, 'dash': [5,3,1,3]},
                'c': {'marker': None, 'dash': [1,3]},
                'm': {'marker': None, 'dash': [5,2,5,2,5,10]},
                'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
                'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
                }

            for line in ax.get_lines():
                origColor = line.get_color()
                line.set_color('black')
                line.set_dashes(COLORMAP[origColor]['dash'])
                line.set_marker(COLORMAP[origColor]['marker'])
                line.set_markersize(MARKERSIZE)

        def setFigLinesBW(fig):
            """
            Take each axes in the figure, and for each line in the axes, make the
            line viewable in black and white.
            """
            for ax in fig.get_axes():
                setAxLinesBW(ax)


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
        for i,xx in enumerate(self.x_sym):
            plt.subplot(int(z1),int(z2),PP)
            PP+=1
            plt.plot(t,xt[:,i])
            plt.xlabel(r'$t$')
            plt.title(r'$'+str(xx)+'(t)$')

        for i,uu in enumerate(self.u_sym):
            plt.subplot(int(z1),int(z2),PP)
            PP+=1
            plt.plot(t,ut[:,i])
            plt.xlabel(r'$t$')
            plt.title(r'$'+str(uu)+'(t)$')

        for hh in self.H:
            plt.subplot(int(z1),int(z2),PP)
            PP+=1
            plt.plot(t,self.H[hh])
            plt.xlabel(r'$t$')
            plt.title(r'$H_'+str(hh+1)+'(t)$')

        setFigLinesBW(fff)

        plt.tight_layout()

        plt.show()


def analyseSystem(ff):
    '''
    This function analyses the system given by the callable vectorfield ``ff(x, u)``
    and returns values for some of the method parameters.


    :param callable ff: Vectorfield of a system of ode`s

    :returns: res -- Dictionary with the results of the system analysis

            ======  =================================
            key     meaning
            ======  =================================
            n       Dimension of the system variables
            m       Dimension of the input variables
            chains  Integrator chains
            ======  =================================

    '''

    res = dict()

    # first, determine system dimensions
    n, m = getDimensions(ff)

    res['n'] = n
    res['m'] = m

    # next, we look for integrator chains
    chains = getIntegChains(ff, (n,m))

    res['chains'] = chains

    # get minimal neccessary number of spline parts for the manipulated variables
    # --> (3.35)      ?????
    #nu = -1
    #...
    #self.su = self.n - 3 + 2*(nu + 1)  ?????


    return res


def getDimensions(ff):
    '''
    This function determines the dimensions of the system and input variables.


    :param callable ff: Vectorfield of a system of ode`s

    :returns: int n -- Dimension of the system variables
    :returns: int m -- Dimension of the input variables
    '''

    i = -1
    j = -1
    found_nm = False

    while not found_nm:
        i += 1
        j += 1
        
        for jj in xrange(j):
            for ii in xrange(i):
                x = np.ones(ii)
                u = np.ones(jj)
                
                try:
                    ff(x,u)
                    found_nm = True
                    n = ii
                    m = jj
                    break
                except:
                    pass

            if found_nm:
                break

    return n, m


def getIntegChains(ff, dim):
    '''This function calls the vectorfield and looks for equations like
    :math:`\\dot{x}_i = x_{i+1}` to find integrator chains


    :param callable ff: Vectorfield of a system of ode`s
    :param tuple dim: (n,m) Dimensions of the system (n) and input (m) variables

    :returns: list chains -- List with objects of found integrator chains
    '''

    log.info( "Looking for integrator chains")

    n, m = dim

    # create symbolic variables to find integrator chains
    x_sym = ([sp.symbols('x%d' % i, type=float) for i in xrange(1,n+1)])
    u_sym = ([sp.symbols('u%d' % i, type=float) for i in xrange(1,m+1)])

    fi = ff(x_sym, u_sym)

    chaindict = {}
    for i in xrange(len(fi)):
        for xx in x_sym:
            if ((fi[i].subs(1.0, 1)) == xx):
                # substitution because of sympy difference betw. 1.0 and 1
                chaindict[xx] = x_sym[i]

        for uu in u_sym:
            if ((fi[i].subs(1.0, 1)) == uu):
                chaindict[uu] = x_sym[i]

    # chaindict looks like this:  {u_1 : x_2, x_4 : x_3, x_2 : x_1}
    # where x_4 = d x_3 / dt and so on

    # find upper ends of integrator chains
    uppers = []
    for vv in chaindict.values():
        if (not chaindict.has_key(vv)):
            uppers.append(vv)

    # create ordered lists that temporarily represent the integrator chains
    tmpchains = []

    # therefor we flip the dictionary to work our way through its keys (former values)
    dictchain = {v:k for k,v in chaindict.items()}

    for var in uppers:
        tmpchain = []
        vv = var
        tmpchain.append(vv)

        while dictchain.has_key(vv):
            vv = dictchain[vv]
            tmpchain.append(vv)

        tmpchains.append(tmpchain)

    # create an integrator chain object for every temporary chain
    chains = []
    for lst in tmpchains:
        chains.append(IntegChain(lst))

    return chains




if __name__ == '__main__':
    from sympy import cos, sin
    from numpy import pi
    import os

    os.environ['SYMPY_USE_CACHE'] = 'no'

    # partiell linearisiertes inverses Pendel [6.1.3]

    calc=True

    def f(x,u):
        x1,x2,x3,x4 = x
        u1, = u
        l = 0.5
        g = 9.81
        ff = np.array([     x2,
                             u1,
                             x4,
                        (1/l)*(g*sin(x3)+u1*cos(x3))])
        return ff

    xa = [  0.0,
            0.0,
            pi,
            0.0]

    xb = [  0.0,
            0.0,
            0.0,
            0.0]

    if(calc):
        a = 0.0
        b = 2.0
        su = 5
        g = [0,0]
        eps = 0.05

        T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, su=su, g=g, eps=eps)

        with log.Timer("Iteration"):
            T.startIteration()
    IPS()
    sys.exit()

