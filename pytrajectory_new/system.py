
# IMPORTS
import numpy as np
import scipy as scp
from scipy import sparse
import sympy as sp

from simulation import Simulator
import auxiliary
import log


# DEBUGGING
DEBUG = True

if DEBUG:
    from IPython import embed as IPS



class ControlSystem(object):
	'''
    here comes the docstring...
    '''
    
    def __init__(self, fnc, a=0.0, b=1.0, xa=[], xb=[], ua=[], ub=[], constraints=None, **kwargs):
        # Save the symbolic vectorfield
        self.ff_sym = ff
        
        # The borders of the considered time interval
        self.a = a
        self.b = b
        
        # Set default values for method parameters
        self.mparam = {'sx' : 5,
                        'su' : 5,
                        'kx' : 2,
                        'delta' : 2,
                        'maxIt' : 10,
                        'eps' : 1e-2,
                        'ierr' : 1e-1,
                        'tol' : 1e-5,
                        'method' : 'leven',
                        'use_chains' : True,
                        'colltype' : 'equidistant',
                        'use_sparse' : True,
                        'sol_steps' : 100}
        
        # Change default values of given kwargs
        for k, v in kwargs.items():
            self.setParam(k, v)
        
        # Analyse the given system to set some parameters
        n, m, x_sym, u_sym, chains, eqind = self.analyseSystem(xa)
        
        # a little check
        if not (len(xa) == len(xb) == n):
            raise ValueError('Dimension mismatch xa, xb')
        
        # Set system dimensions
        self.n = n
        self.m = m
        
        # Set symbols for state and input variables
        self.x_sym = x_sym
        self.u_sym = u_sym
        
        # Set integrator chains and equations that have to be solved
        self.chains = chains
        self.eqind = eqind
        
        # Transform lists of boundary values into dictionaries
        xa = dict()
        xb = dict()
        ua = dict()
        ub = dict()
        
        for i, xx in enumerate(self.x_sym):
            xa[xx] = xa[i]
            xb[xx] = xb[i]
        
        for i, uu in enumerate(self.u_sym):
            try:
                ua[uu] = ua[i]
            except:
                ua[uu] = None
            
            try:
                ub[uu] = ub[i]
            except:
                ub[uu] = None
        
        self.xa = xa
        self.xb = xb
        self.ua = ua
        self.ub = ub
        
        # Handle system constraints if there are any
        if constraints:
            self.constraints = constraints
            
            # transform the constrained vectorfield into an unconstrained one
            ff_sym, xa, xb, orig_backup = self.unconstrain()
            
            self.ff_sym = ff_sym
            self.xa = xa
            self.xb = xb
            self.orig_backup = orig_backup
            
            # we cannot make use of an integrator chain
            # if it contains a constrained variable
            self.mparam['use_chains'] = False
            # TODO: implement it so that just those chains are not use 
            #       which actually contain a constrained variable
        
        # Now we transform the symbolic function of the vectorfield to
        # a numeric one for faster evaluation
        self.ff = sym2num_vectorfield(self.ff_sym, self.x_sym, self.u_sym)
        
        # Create trajectories
        self.trajectories = Trajectory(self)
        
        # Reset iteration number
        self.nIt = 0
        
        # We didn't really do anything yet, so this should be false
        self.reached_accuracy = False

        # and likewise this should not be existent yet
        self.sol = None
    
    
    def setParam(self, param='', val=None):
        '''
        Method to assign value :attr:`val` to method parameter :attr:`param`.
        (mainly for didactic purpose)

        Parameters
        ----------

        param : str
            Parameter of which to alter the value.

        val : ???
            New value for the passed parameter.
        '''
        
        # check if current and new value have the same type
        # --> should they always?
        assert type(val) == type(self.mparam[param])
        
        self.mparam[param] = val
    
    
    def analyseSystem(self, xa):
        '''
        Analyse the system structure and set values for some of the method parameters.

        By now, this method determines the number of state and input variables, creates
        sympy.symbols for them and searches for integrator chains.
        
        Parameters
        ----------
        
        xa : list
            Initial values of the state variables (for determining the system dimensions)
        
        Returns
        -------
        
        int
            System state dimension.
        
        int
            Input dimension.
        
        list
            List of sympy.symbols for state variables
        
        list
            List of sympy.symbols for input variables
        
        list
            Found integrator chains of the system.
        
        list
            Indices of the equations that have to be solved using collocation.
        '''

        log.info("  Analysing System Structure", verb=2)
        
        # first, determine system dimensions
        log.info("    Determine system/input dimensions", verb=3)
        
        # the number of system variables can be determined via the length
        # of the boundary value lists
        n = len(xa)
        
        # now we want to determine the input dimension
        # therefore we iteratively increase the inputs dimension and try to call
        # the vectorfield
        found_m = False
        j = 0
        x = np.ones(n)
        while not found_m:
            u = np.ones(j)
            try:
                self.ff_sym(x, u)
                # if no ValueError is raised j is the dimension of the inputs
                m = j
                found_m = True
            except ValueError:
                # unpacking error inside ff_sym
                # (that means the dimensions don't match)
                j += 1
        
        log.info("      --> system: %d"%n, verb=3)
        log.info("      --> input : %d"%m, verb=3)

        # next, we look for integrator chains
        log.info("    Looking for integrator chains", verb=3)

        # create symbolic variables to find integrator chains
        x_sym = ([sp.symbols('x%d' % k, type=float) for k in xrange(1,n+1)])
        u_sym = ([sp.symbols('u%d' % k, type=float) for k in xrange(1,m+1)])

        fi = self.ff_sym(x_sym, u_sym)

        chains, eqind = auxiliary.findIntegratorChains(fi, x_sym)
        
        # get minimal neccessary number of spline parts
        # for the manipulated variables
        # TODO: implement this!?
        # --> (3.35)      ?????
        #nu = -1
        #...
        #self.su = self.n - 3 + 2*(nu + 1)  ?????
        
        return n, m, x_sym, u_sym, chains, eqind
    
    
    def unconstrain(self):
        '''
        This method is used to enable compliance with the desired box constraints.
        It transforms the vectorfield by projecting the constrained state variables on
        new unconstrained ones.
        '''
        
        # make some stuff local
        ff = sp.Matrix(self.ff_sym(self.x_sym, self.u_sym))
        xa = self.xa
        xb = self.xb
        x_sym = self.x_sym
        
        # First, we backup all things that will be influenced in some way
        #
        # backup original state variables and their boundary values
        x_sym_orig = 1*x_sym
        xa_orig = xa.copy()
        xb_orig = xb.copy()
        
        # backup symbolic vectorfield function
        ff_sym_orig = self.ff_sym
        
        # create a numeric vectorfield function of the original vectorfield
        # and back it up (will be used in simulation step of the main iteration)
        ff_num_orig = sym2num_vectorfield(ff_sym_orig, x_sym_orig, self.u_sym)
        
        # Now we can handle the constraints by projecting the constrained state variables
        # on new unconstrained variables using saturation functions
        for k, v in self.constraints.items():
            # check if boundary values are within saturation limits
            if not ( v[0] < xa[x_sym[k]] < v[1] ) or not ( v[0] < xb[x_sym[k]] < v[1] ):
                log.error('Boundary values have to be strictly within the saturation limits!')
                log.info('Please have a look at the documentation, \
                          especially the example of the constrained double intgrator.')
                raise ValueError('Boundary values have to be strictly within the saturation limits!')
            
            # replace constrained state variable with new unconstrained one
            x_sym[k] = sp.Symbol('y%d'%(k+1))
            
            # calculate saturation function expression and its derivative
            yk = x_sym[k]
            m = 4.0/(v[1] - v[0])
            psi = v[1] - (v[1]-v[0])/(1.0+sp.exp(m*yk))
            #dpsi = ((v[1]-v[0])*m*sp.exp(m*yk))/(1.0+sp.exp(m*yk))**2
            dpsi = (4.0*sp.exp(m*yk))/(1.0+sp.exp(m*yk))**2
            
            # replace constrained variables in vectorfield with saturation expression
            # x(t) = psi(y(t))
            ff = ff.replace(x_sym_orig[k], psi)
            
            # update vectorfield to represent differential equation for new
            # unconstrained state variable
            #
            # d/dt x(t) = (d/dy psi(y(t))) * d/dt y(t)
            # <==> d/dt y(t) = d/dt x(t) / (d/dy psi(y(t)))
            ff[k] = ff[k] / dpsi
            
            # replace key of constrained variable in dictionaries for boundary values
            # with new symbol for the unconstrained variable
            xk = x_sym_orig[k]
            xa[yk] = xa.pop(xk)
            xb[yk] = xb.pop(xk)
            
            # update boundary values for new unconstrained variable
            wa = xa[yk]
            xa[yk] = (1.0/m)*np.log( (wa-v[0])/(v[1]-wa) )
            wb = xb[yk]
            xb[yk] = (1.0/m)*np.log( (wb-v[0])/(v[1]-wb) )
        
        # create a callable function for the new symbolic vectorfield
        if ff.T == ff.vec():
            ff = ff.tolist()[0]
        else:
            ff = ff.T.tolist()[0]
        
        _ff_sym = sp.lambdify(x_sym+self.u_sym, ff, modules='sympy')
        
        def ff_sym(x,u):
            xu = np.hstack((x,u))
            return np.array(_ff_sym(*xu))
        
        # backup the original ones, as well as some other stuff
        orig_backup = dict()
        orig_backup = {'xa' : xa_orig, 'xb' : xb_orig,
                        'ff_sym' : ff_sym_orig, 'ff_num' : ff_num_orig,
                        'x_sym' : x_sym_orig}
        
        return ff_sym, xa, xb, orig_backup
    
    
    def solveControlProblem(self):
        '''
        here comes the docstring...
        '''
        
        # do the first step
        self.iterate()
        
        # check if desired accuracy is already reached
        self.trajectories.checkAccuracy()
        
        # this was the first iteration
        # now we are getting into the loop
        while not self.reached_accuracy and self.nIt < self.mparam['maxIt']:
            # raise the number of spline parts
            self.mparam['sx'] = int(round(self.mparam['kx']*self.mparam['sx']))
            
            if self.nIt == 1:
                log.info("2nd Iteration: %d spline parts"%self.mparam['sx'], verb=1)
            elif self.nIt == 2:
                log.info("3rd Iteration: %d spline parts"%self.mparam['sx'], verb=1)
            elif self.nIt >= 3:
                log.info("%dth Iteration: %d spline parts"%(self.nIt+1, self.mparam['sx']), verb=1)

            # store the old spline to calculate the guess later
            self.trajectories.old_splines = self.trajectories.splines

            # start next iteration step
            self.iterate()

            # check if desired accuracy is reached
            self.reached_accuracy = self.trajectories.checkAccuracy()
            
        # clear workspace
        self.clear()
        
        # as a last we, if there were any constraints to be taken care of,
        # we project the unconstrained variables back on the original
        # constrained ones
        if self.constraints:
            self.constrain()
        
        log.log_off()
        
        # return the found solution functions
        return self.trajectories.x, self.trajectories.u
    
    
    def iterate(self):
        '''
        here comes the docstring...
        '''
        
        # Increase iteration number
        self.nIt += 1
        
        # Initialise the spline function objects
        self.trajectories.initSplines()
        
        # Build the collocation equations system
        G, DG = self.eqs.build()
        
        # Get a initial value (guess)
        self.eqs.getGuess()
        
        # Solve the collocation equation system
        self.eqs.solve(D, DG)
        
        # Set the found solution
        self.trajectories.setCoeffs()
        
        # Solve the resulting initial value problem
        self.simulate()
    
    
    def simulate(self):
        '''
        This method is used to solve the initial value problem.
        '''

        log.info("  Solving Initial Value Problem", verb=2)
        
        # calulate simulation time
        T = self.b - self.a
        
        # get list of start values
        start = []
        
        if self.constraints:
            start_dict = self.orig_backup['xa']
            x_vars = self.orig_backup['x_sym']
            ff = self.orig_backup['ff_num']
        else:
            start_dict = self.xa
            x_vars = self.x_sym
            ff = self.ff
        
        for x in x_vars:
            start.append(start_dict[x])
        
        # create simulation object
        S = Simulator(ff, T, start, self.trajectories.u)
        
        log.info("    start: %s"%str(start), verb=2)
        
        # start forward simulation
        self.sim = S.simulate()
        















