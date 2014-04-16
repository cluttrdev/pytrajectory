import os
import sys

import numpy as np
import sympy as sp
import scipy as scp
from scipy import sparse

import pickle

from spline import CubicSpline, fdiff
from solver import Solver
from simulation import Simulation
from utilities import IntegChain

#####
#NEW
from utilities import blockdiag as bdiag
from utilities import plot as plotsim
#####

import log
#from log import IPS
from time import time

from IPython import embed as IPS


class Trajectory():
    '''
    Base class of the PyTrajectory project.
    
    
    Parameters
    ----------
    
    ff :  callable
        Vectorfield (rhs) of the control system
    a : float
        Left border 
    b : float
        Right border
    xa : list
        Boundary values at the left border
    xb : list
        Boundary values at the right border
    g : list
        Boundary values of the input variables
    sx : int
        Initial number of spline parts for the system variables
    su : int
        Initial number of spline parts for the input variables
    kx : int
        Factor for raising the number of spline parts for the system variables
    delta : int
        Constant for calculation of collocation points
    maxIt : int
        Maximum number of iterations
    eps : float
        Tolerance for the solution of the initial value problem
    tol : float
        Tolerance for the solver of the equation system
    algo : str
        Solver to use
    use_chains : bool
        Whether or not to use integrator chains
    '''
    
    def __init__(self, ff, a=0.0, b=1.0, xa=None, xb=None, g=None, sx=5, su=5, kx=2,
                delta=2, maxIt=10, eps=1e-2, tol=1e-5, algo='leven', use_chains=True):
        
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
        self.analyseSystem()
        
        # a little check
        if not (len(xa) == len(xb) == self.n):
            raise ValueError, 'Dimension mismatch xa,xb'
        
        # type of collocation points to use
        self.colltype = 'equidistant'
        #self.colltype = 'chebychev'
        
        # whether or not to use sparse matrices
        self.use_sparse = True
        
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
        self.x_sym = ([sp.symbols('x%d' % i, type=float) for i in xrange(1,self.n+1)])
        self.u_sym = ([sp.symbols('u%d' % i, type=float) for i in xrange(1,self.m+1)])
        
        # transform symbolic ff_sym to numeric ff_num for faster evaluation
        F = sp.Matrix(ff(self.x_sym,self.u_sym))
        _ff_num = sp.lambdify(self.x_sym+self.u_sym, F, modules='numpy')
        
        def ff_num(x, u):
            xu = np.hstack((x, u))
            return np.array(_ff_num(*xu)).squeeze()
        
        self.ff = ff_num

        # dictionaries for boundary conditions
        self.xa = dict()
        self.xb = dict()
        
        for i,xx in enumerate(self.x_sym):
            self.xa[xx] = xa[i]
            self.xb[xx] = xb[i]

        # set logfile
        #log.set_file()
        
        # just for me
        print np.__version__
        print sp.__version__
        print scp.__version__
    
    
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
        # --> lower ends of integrator chains
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
        
        # check if desired accuracy is already reached
        self.checkAccuracy()

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

            # check if desired accuracy is reached
            self.checkAccuracy()
        
        # clear workspace
        self.clear()
    
    
    def analyseSystem(self):
        '''
        Analyses the systems structure and sets values for some of the method parameters.
        '''
        
        log.info( 40*"#")
        log.info("####   Analysing System Strucutre   ####")
        log.info( 40*"#")
        
        ###
        # first, determine system dimensions
        ###
        log.info("Determine system/input dimensions")
        i = -1
        found_nm = False
        
        while not found_nm:
            # iteratively increase system and input dimensions and try to call
            # symbolic vectorfield ff_sym with i/j-dimensional vectors
            i += 1
            
            for j in xrange(i):
                x = np.ones(i)
                u = np.ones(j)
                
                try:
                    self.ff_sym(x, u)
                    # if no ValueError is raised, i is the system dimension
                    # and j is the dimension of the inputs
                    found_nm = True
                    n = i
                    m = j
                    break
                except ValueError:
                    # unpacking error inside ff_sym
                    pass
                    #~ if j == 0:
                        #~ try:
                            #~ self.ff_sym(x)
                            #~ found_nm = True
                            #~ n = i
                            #~ m = j
                            #~ break
                        #~ except:
                            #~ pass
        
        self.n = n
        self.m = m
        
        log.info("---> system: %d"%n)
        log.info("---> input : %d"%m)
        
        ###
        # next, we look for integrator chains
        ###
        log.info("Look for integrator chains")
        
        # create symbolic variables to find integrator chains
        x_sym = ([sp.symbols('x%d' % i, type=float) for i in xrange(1,n+1)])
        u_sym = ([sp.symbols('u%d' % i, type=float) for i in xrange(1,m+1)])
        
        fi = self.ff_sym(x_sym, u_sym)
    
        chaindict = {}
        for i in xrange(len(fi)):
            # substitution because of sympy difference betw. 1.0 and 1
            if isinstance(fi[i], sp.Basic):
                fi[i] = fi[i].subs(1.0, 1)
            
            for xx in x_sym:
                if fi[i] == xx:
                    chaindict[xx] = x_sym[i]
            
            for uu in u_sym:
                if fi[i] == uu:
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
            ic = IntegChain(lst)
            chains.append(ic)
            log.info("---> found: " + str(ic))
        
        self.chains = chains
        
        # get minimal neccessary number of spline parts for the manipulated variables
        # --> (3.35)      ?????
        #IPS()
        #nu = -1
        #...
        #self.su = self.n - 3 + 2*(nu + 1)  ?????
    
    
    def setParam(self, param='', val=None):
        '''
        Method to assign value :attr:`val`to method parameter :attr:`param`.
        
        
        Parameters
        ----------
        
        param : str
            Parameter of which to alter the value
        
        val : ???
            New value for the passed parameter
        '''
        
        #if param and val:
        #    exec('self.%s = %s'%(param, str(val)))
        #elif val and not param:
        #    log.warn('No method parameter given to assign value %s to!'%str(val))
        #elif param and not val:
        #    log.warn('No value passed to assign to method parameter %s!'%param)
        #else:
        #    pass
        exec('self.%s = %s'%(param, str(val)))
    
    
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
                    gpts = np.linspace(self.a,self.b,(nn+1),endpoint = False)[1:]

                    # evaluate the old and new spline at all points in gpts
                    #   they should be equal in these points

                    OLD = [None]*len(gpts)
                    NEW = [None]*len(gpts)
                    NEW_abs = [None]*len(gpts)
                    
                    for i, p in enumerate(gpts):
                        OLD[i] = old_splines[k].f(p)
                        NEW[i], NEW_abs[i] = new_splines[k].f(p)
                    
                    OLD = np.array(OLD)
                    NEW = np.array(NEW)
                    NEW_abs = np.array(NEW_abs)

                    TT = np.linalg.solve(NEW,OLD-NEW_abs)

                    guess = np.hstack((guess,TT))
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
                splines[upper] = CubicSpline(self.a,self.b,n=self.sx,bc=[self.xa[upper],self.xb[upper]],steady=False,tag=upper.name)
                splines[upper].type = 'x'
            elif ic.lower.name.startswith('u'):
                splines[upper] = CubicSpline(self.a,self.b,n=self.su,bc=self.g,steady=False,tag=upper.name)
                splines[upper].type = 'u'
            
            for i,elem in enumerate(ic.elements):
                if elem in self.u_sym:
                    if (i == 0):
                        u_fnc[elem] = splines[upper].f
                    if (i == 1):
                        u_fnc[elem] = splines[upper].df
                    if (i == 2):
                        u_fnc[elem] = splines[upper].ddf
                elif elem in self.x_sym:
                    if (i == 0):
                        splines[upper].bc = [self.xa[elem],self.xb[elem]]
                        if ((self.g != None) and (splines[upper].type == 'u')):
                            splines[upper].bcd = self.g
                        x_fnc[elem] = splines[upper].f
                    if (i == 1):
                        splines[upper].bcd = [self.xa[elem],self.xb[elem]]
                        if ((self.g != None) and (splines[upper].type == 'u')):
                            splines[upper].bcdd = self.g
                        x_fnc[elem] = splines[upper].df
                    if (i == 2):
                        splines[upper].bcdd = [self.xa[elem],self.xb[elem]]
                        x_fnc[elem] = splines[upper].ddf
        
        # now handle the variables which are not part of any chain
        for xx in self.x_sym:
            if (not x_fnc.has_key(xx)):
                splines[xx] = CubicSpline(self.a,self.b,n=self.sx,bc=[self.xa[xx],self.xb[xx]],steady=False,tag=str(xx))
                splines[xx].type = 'x'
                x_fnc[xx] = splines[xx].f

        for uu in self.u_sym:
            if (not u_fnc.has_key(uu)):
                splines[uu] = CubicSpline(self.a,self.b,n=self.su,bc=self.g,steady=False,tag=str(uu))
                splines[uu].type = 'u'
                u_fnc[uu] = splines[uu].f

        # solve smoothness conditions of each spline
        for ss in splines:
            with log.Timer("makesteady()"):
                splines[ss].makesteady()

        for xx in self.x_sym:
            dx_fnc[xx] = fdiff(x_fnc[xx])

        indep_coeffs= dict()
        for ss in splines:
            indep_coeffs[ss] = splines[ss].c_indep

        self.indep_coeffs = indep_coeffs

        self.splines = splines
        self.x_fnc = x_fnc
        self.u_fnc = u_fnc
        self.dx_fnc = dx_fnc


    def buildEQS(self):
        '''
        Builds the collocation equation system.
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
        if self.colltype == 'equidistant':
            # get equidistant collocation points
            cpts = np.linspace(a,b,(self.sx*delta+1),endpoint=True)
        elif self.colltype == 'chebychev':
            # determine rank of chebychev polynomial of which to calculate zero points
            nc = int(self.sx*delta - 1)
            
            # calculate zero points of chebychev polynomial --> in [-1,1]
            cheb_cpts = [np.cos( (2.0*i+1)/(2*(nc+1)) * np.pi) for i in xrange(nc)]
            cheb_cpts.sort()
            
            # transfer chebychev knots from [-1,1] to our interval [a,b]
            a = self.a
            b = self.b
            chpts = [a + (b-a)/2.0 * (chp + 1) for chp in cheb_cpts]
            
            # add left and right borders
            cpts = np.hstack((a, chpts, b))
        else:
            log.warn('Unknown type of collocation points.')
            log.warn('--> will use equidistant points!')
            cpts = np.linspace(a,b,(self.sx*delta+1),endpoint=True)
        
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
        for i,p in enumerate(cpts):
            for xx in x_sym:
                mx = np.zeros(c_len)
                mdx = np.zeros(c_len)

                i,j= indic[xx]

                mx[i:j], Mx_abs[eqx] = x_fnc[xx](p)
                mdx[i:j], Mdx_abs[eqx] = dx_fnc[xx](p)

                Mx[eqx] = mx
                Mdx[eqx] = mdx
                eqx += 1

            for uu in u_sym:
                mu = np.zeros(c_len)

                i,j = indic[uu]

                mu[i:j], Mu_abs[equ] = u_fnc[uu](p)

                Mu[equ] = mu
                equ += 1

        self.Mx = np.array(Mx)
        self.Mx_abs = np.array(Mx_abs)
        self.Mdx = np.array(Mdx)
        self.Mdx_abs = np.array(Mdx_abs)
        self.Mu = np.array(Mu)
        self.Mu_abs = np.array(Mu_abs)

        # for creation of the jacobian matrix
        f = self.ff_sym(x_sym,u_sym)
        Df_mat = sp.Matrix(f).jacobian(self.x_sym+self.u_sym)
        
        self.Df = sp.lambdify(self.x_sym+self.u_sym, Df_mat, modules='numpy')
        
        # the following would be created with every call to self.DG but it is possible to 
        # only do it once
        self.DdX = self.Mdx.reshape((len(self.cpts),-1,len(self.c_list)))[:,self.eqind,:]
        
        ##########
        # NEW
        J_XU = []
        x_len = len(self.x_sym)
        u_len = len(self.u_sym)
        
        for i in xrange(len(cpts)):
            J_XU.append(np.vstack(( self.Mx[x_len*i:x_len*(i+1)], self.Mu[u_len*i:u_len*(i+1)] )))
        
        #self.J_XU = np.array(J_XU)
        self.J_XU = J_XU
        J_XU2 = np.vstack(np.array(J_XU))
        self.J_XU2 = sparse.csr_matrix(J_XU2)
        ##########
        
        if self.use_sparse:
            self.Mx = sparse.csr_matrix(self.Mx)
            self.Mx_abs = sparse.csr_matrix(self.Mx_abs)
            self.Mdx = sparse.csr_matrix(self.Mdx)
            self.Mdx_abs = sparse.csr_matrix(self.Mdx_abs)
            self.Mu = sparse.csr_matrix(self.Mu)
            self.Mu_abs = sparse.csr_matrix(self.Mu_abs)
            
            self.DdX = sparse.csr_matrix(np.vstack(self.DdX))
    

    def solve(self):
        '''
        This method is used to solve the collocation equation system.
        '''
        
        log.info( 40*"#")
        log.info("#####  Solving the equation system  ####")
        log.info( 40*"#")

        # create our solver
        solver = Solver(self.G, self.DG, self.guess, tol= self.tol,
                        maxx= 20, algo=self.algo)

        # solve the equation system
        self.sol = solver.solve()


    def G(self, c):
        '''
        Returns the collocation system evaluated with numeric values for the independent parameters.
        '''
        
        ff = self.ff
        eqind = self.eqind

        x_len = len(self.x_sym)
        u_len = len(self.u_sym)

        X = self.Mx.dot(c) + self.Mx_abs
        U = self.Mu.dot(c) + self.Mu_abs

        X = np.array(X).reshape((-1,x_len))
        U = np.array(U).reshape((-1,u_len))

        # evaluate system equations and select those related
        # to lower ends of integrator chains (via eqind)
        # other equations need not to be solved
        F = np.array([ff(x,u) for x,u in zip(X,U)], dtype=float).squeeze()[:,eqind]
        
        dX = self.Mdx.dot(c) + self.Mdx_abs
        dX = np.array(dX).reshape((-1,x_len))[:,eqind]

        G = F-dX

        return G.flatten()


    def DG(self, c):
        '''
        Returns the Jacobian matrix of the collocation system w.r.t. the independent parameters.
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
        
        ##########
        # The following two loops are time consuming
        #
        
        # construct jacobian of rhs w.r.t. indep coeffs
        DF_blocks = []
        for x,u in zip(X,U):
            # get one row of U and X respectively
            tmp_xu = np.hstack((x,u))
        
            # evaluate jacobian at current collocation point
            DF_blocks.append(Df(*tmp_xu))
        
        if 0:
            ###############################################
            # OLD - working
            ###############################################
            DF = []
            #for i,df in enumerate(DF_blocks):
            #    # np.vstack is done in every call --> do it once in buildEQS...???
            #    J_XU = np.vstack(( self.Mx[x_len*i:x_len*(i+1)], self.Mu[u_len*i:u_len*(i+1)] ))
            #    res = np.dot(df,J_XU)
            for i in xrange(len(DF_blocks)):
                res = np.dot(DF_blocks[i], self.J_XU[i])
                assert res.shape == (x_len,len(self.c_list))
                DF.append(res)
            #IPS()
            DF = np.array(DF)[:,eqind,:]
            # 1st index : collocation point
            # 2nd index : equations that have to be solved --> end of an integrator chain
            # 3rd index : component of c
        else:
            ###############################################
            # NEW - experimental
            ###############################################
            block_DF = bdiag(np.vstack(np.array(DF_blocks)), (x_len, x_len+u_len),True)
            J_XU = self.J_XU2
            DF2 = block_DF.dot(J_XU)
            #DF2.reshape((-1,x_len,len(c))) # --> to many reshape dimensions for sparse matrix
            DF2 = DF2.toarray().reshape((-1,x_len,len(c)))
            DF = DF2[:,eqind,:]
        
        
        # now compute jacobian of x_dot w.r.t. to indep coeffs
        # --> see buildEQS()
        #DdX = self.Mdx.reshape((len(self.cpts),-1,len(self.c_list)))[:,eqind,:]
        DdX = self.DdX
        
        # stack matrices in vertical direction
        #DG = np.vstack(DF) - np.vstack(DdX)
        DG = np.vstack(DF) - DdX
        
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

        # yet another dictionary for solution and coeffs
        coeffs_sol = dict()

        # used for indexing
        i = 0
        j = 0
        
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
        
        # start forward simulation
        self.sim = S.simulate()

        # calculate the error functions H_i(t)
        H = dict()
        
        error = []
        for t in self.sim[0]:
            xe = self.x(t)
            ue = self.u(t)
            
            ffe = self.ff(xe, ue)
            dxe = self.dx(t)
            
            error.append(ffe - dxe)
        error = np.array(error)
        
        for i in self.eqind:
            H[i] = error[:,i]
        
        self.H = H
    
    
    def checkAccuracy(self):
        '''
        Checks whether desired accuracy for the boundary values was reached.
        '''
        
        # this is the solution of the simulation
        xt = self.sim[1]

        # what is the error
        log.info(40*"-")
        log.info("Ending up with:\t Should Be: \t Difference:")
        
        err = np.empty(self.n)
        for i, xx in enumerate(self.x_sym):
            err[i] = abs(self.xb[xx] - xt[-1:][0][i])
            log.info(str(xx)+" : %f \t %f \t %f"%(xt[-1][i], self.xb[xx], err[i]))
        
        log.info(40*"-")
        
        self.err = err
        
        errmax = max(self.err)
        log.info("--> reached desired accuracy: "+str(errmax <= self.eps))
    
    
    def x(self, t):
        '''
        This function returns the system state at a given (time-) point :attr:`t`.
        '''
        return np.array([self.x_fnc[xx](t) for xx in self.x_sym])
    

    def u(self, t):
        '''
        This function returns the inputs state at a given (time-) point :attr:`t`.
        '''
        return np.array([self.u_fnc[uu](t) for uu in self.u_sym])
    
    
    def dx(self, t):
        '''
        This function returns the left hand sites state at a given (time-) point :attr:`t`.
        '''
        return np.array([self.dx_fnc[xx](t) for xx in self.x_sym])
    
    
    def plot(self):
        '''
        Just calls :func:`plot` function from :mod:`utilities`
        '''
        plotsim(self.sim, self.H)
    
    
    def save(self):
        '''
        Save system data, callable solution functions and simulation results.
        '''
        
        save = dict()
        
        # system data
        save['ff_sym'] = self.ff_sym
        save['ff_num'] = self.ff_num
        save['a'] = self.a
        save['b'] = self.b
        
        # boundary values
        save['xa'] = self.xa
        save['xb'] = self.xb
        
        # solution functions
        save['x'] = self.x
        save['u'] = self.u
        save['dx'] = self.dx
        
        # simulation resutls
        save['sim'] = self.sim
    
    
    def clear(self):
        pass


if __name__ == '__main__':
    from sympy import cos, sin
    from numpy import pi
    
    # partiell linearisiertes inverses Pendel [6.1.3]

    calc = True

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
        sx = 5
        su = 5
        kx = 5
        maxIt  = 5
        g = [0,0]
        eps = 0.05
        use_chains = True

        T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, sx=sx, su=su, kx=kx,
                        maxIt=maxIt, g=g, eps=eps, use_chains=use_chains)

        with log.Timer("Iteration"):
            T.startIteration()
        
        IPS()

