
import numpy as np
import sympy as sp
import scipy as scp
from scipy import sparse
import pickle

from spline import CubicSpline, fdiff
from solver import Solver
from simulation import Simulation
import utilities

from utilities import IntegChain, plotsim
from log import logging, Timer


def sym2num_vectorfield(f_sym, x_sym, u_sym):
    '''
    This function takes a callable vectorfield of a control system that is to be evaluated with symbols
    for the state and input variables and returns a corresponding function that can be evaluated with
    numeric values for these variables.
    
    Parameters
    ----------
    
    f_sym : callable
        The callable ("symbolic") vectorfield of the control system.
    
    x_sym : iterable
        The symbols for the state variables of the control system.
    
    u_sym : iterable
        The symbols for the input variables of the control system.
    
    Returns
    -------
    
    callable
        The callable ("numeric") vectorfield of the control system.
    '''
    
    # get a sympy.Matrix representation of the vectorfield
    F = sp.Matrix(f_sym(x_sym, u_sym))
    if F.T == F.vec():
        F = F.tolist()[0]
    else:
        F = F.T.tolist()[0]
    
    # Use lambdify to replace sympy functions in the vectorfield with
    # numpy equivalents
    _f_num = sp.lambdify(x_sym + u_sym, F, modules='numpy')
    
    # Create a wrapper as the actual function due to the behaviour
    # of lambdify()
    def f_num(x, u):
        xu = np.hstack((x, u))
        return np.array(_f_num(*xu))
    
    return f_num


def saturation_functions(y_fnc, dy_fnc, y0, y1):
    '''
    Creates callable saturation function and its first derivative to project 
    the solution found for an unconstrained state variable back on the original
    constrained one.
    
    For more information, please have a look at :ref:`handling_constraints`.
    
    Parameters
    ----------
    
    y_fnc : callable
        The calculated solution function for an unconstrained variable.
    
    dy_fnc : callable
        The first derivative of the unconstrained solution function.
    
    y0 : float
        Lower saturation limit.
    
    y1 : float
        Upper saturation limit.
    
    Returns
    -------
    
    callable
        A callable of a saturation function applied to a calculated solution
        for an unconstrained state variable.
    
    callable
        A callable for the first derivative of a saturation function applied 
        to a calculated solution for an unconstrained state variable.
    '''
    
    # Calculate the parameter m such that the slope of the saturation function
    # at t = 0 becomes 1
    m = 4.0/(y1-y0)
    
    # this is the saturation function
    def psi_y(t):
        y = y_fnc(t)
        return y1 - (y1-y0)/(1.0+np.exp(m*y))
    
    # and this its first derivative
    def dpsi_dy(t):
        y = y_fnc(t)
        dy = dy_fnc(t)
        return dy * (4.0*np.exp(m*y))/(1.0+np.exp(m*y))**2
    
    return psi_y, dpsi_dy


def consistency_error(I, x_fnc, u_fnc, dx_fnc, ff_fnc, npts=500,return_error_array=False):
    '''
    
    Parameters
    ----------
    
    I : tuple
        The considered time interval.
    
    x_fnc : callable
        A function for the state variables.
    
    u_fnc : callable
        A function for the input variables.
    
    dx_fnc : callable
        A function for the first derivatives of the state variables.
    
    ff_fnc : callable
        A function for the vectorfield of the control system.
    
    npts : int
        Number of point to determine the error at.
    
    return_error_array : bool
        Whether or not to return the calculated errors (mainly for plotting).
    
    Returns
    -------
    
    float
        The maximum error between the systems dynamic and its approximation.
    
    numpy.ndarray
        An array with all errors calculated on the interval.
    '''
    
    # get some test points to calculate the error at
    tt = np.linspace(I[0], I[1], npts, endpoint=True)
    
    error = []
    for t in tt:
        x = x_fnc(t)
        u = u_fnc(t)
        
        ff = ff_fnc(x, u)
        dx = dx_fnc(t)
        
        error.append(ff - dx)
    error = np.array(error)
    max_con_err = error.max()
    
    if return_error_array:
        return max_con_err, error
    else:
        return max_con_err


class Trajectory():
    '''
    Base class of the PyTrajectory project.

    Trajectory manages everything from analysing the given system over
    initialising the spline functions, setting up and solving the collocation
    equation system up to the simulation of the resulting initial value problem.

    After the iteration has finished, it provides access to callable functions
    for the system and input variables as well as some capabilities for
    visualising the systems dynamic.

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
    
    ua : list
        Boundary values of the input variables at left border
    
    ub : list
        Boundary values of the input variables at right border
    
    constraints : dict
        Box-constraints of the state variables
    
    Attributes
    ----------
    
    mparam : dict
        Dictionary with method parameters
        
        ==========  =============   =======================================================
        key         default value   meaning
        ==========  =============   =======================================================
        sx          5               Initial number of spline parts for the system variables
        su          5               Initial number of spline parts for the input variables
        kx          2               Factor for raising the number of spline parts
        delta       2               Constant for calculation of collocation points
        maxIt       10               Maximum number of iteration steps
        eps         1e-2            Tolerance for the solution of the initial value problem
        ierr        1e-1            Tolerance for the error on the whole interval
        tol         1e-5            Tolerance for the solver of the equation system
        method      'leven'         The solver algorithm to use
        use_chains  True            Whether or not to use integrator chains
        colltype    'equidistant'   The type of the collocation points
        use_sparse  True            Whether or not to use sparse matrices
        sol_steps   100             Maximum number of iteration steps for the eqs solver
        ==========  =============   =======================================================
    
    '''
    
    def __init__(self, ff, a=0.0, b=1.0, xa=[], xb=[], ua=[], ub=[], constraints=None, **kwargs):
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
        
        # Initialise dimensions --> they will be set afterwards
        #self.n = 0
        #self.m = 0
        
        # Analyse the given system to set some parameters
        self.analyseSystem(xa)
        
        # A little check
        if not (len(xa) == len(xb) == self.n):
            raise ValueError('Dimension mismatch xa,xb')

        # Reset iteration number
        self.nIt = 0

        # Dictionary for spline objects
        #   key: variable   value: CubicSpline-object
        self.splines = dict()

        # Dictionaries for callable (solution) functions
        self.x_fnc = dict()
        self.u_fnc = dict()
        self.dx_fnc = dict()

        # Create symbolic variables -> this is now done by analyseSystem()
        #self.x_sym = ([sp.symbols('x%d' % i, type=float) for i in xrange(1,self.n+1)])
        #self.u_sym = ([sp.symbols('u%d' % i, type=float) for i in xrange(1,self.m+1)])
        
        # Dictionaries for boundary conditions
        self.xa = dict()
        self.xb = dict()
        
        for i, xx in enumerate(self.x_sym):
            self.xa[xx] = xa[i]
            self.xb[xx] = xb[i]
        
        self.ua = dict()
        self.ub = dict()
        
        for i, uu in enumerate(self.u_sym):
            try:
                self.ua[uu] = ua[i]
            except:
                self.ua[uu] = None
            
            try:
                self.ub[uu] = ub[i]
            except:
                self.ub[uu] = None
        
        # now we handle system constraints if there are any
        self.constraints = constraints
        if self.constraints:
            # transform the constrained vectorfield into an unconstrained one
            self.unconstrain()
            
            # we cannot make use of an integrator chain
            # if it contains a constrained variable
            self.mparam['use_chains'] = False
            # TODO: implement it so that just those chains are not use 
            #       which actually contain a constrained variable
        
        # Now we transform the symbolic function of the vectorfield to
        # a numeric one for faster evaluation
        self.ff = sym2num_vectorfield(self.ff_sym, self.x_sym, self.u_sym)
        
        # initialise dummy procedures for the later equation system and its jecobian
        # --> they will be defined in self.buildEQS()
        self.G = lambda c : 'NotImplemented'   #raise NotImplementedError
        self.DG = lambda c : 'NotImplemented'  #raise NotImplementedError
        
        # We didn't really do anything yet, so this should be false
        self.reached_accuracy = False

        # and likewise this should not be existent yet
        self.sol = None

        
    def analyseSystem(self, xa):
        '''
        Analyse the system structure and set values for some of the method parameters.

        By now, this method determines the number of state and input variables, creates
        sympy.symbols for them and searches for integrator chains.
        
        Parameters
        ----------
        
        xa : list
            Initial values of the state variables (for determining the system dimensions)
        '''

        logging.debug("Analysing System Structure")
        
        # first, determine system dimensions
        logging.debug("Determine system/input dimensions")
        
        # the number of input variables can be determined via the length
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
                
        # set system dimensions
        self.n = n
        self.m = m

        logging.debug("--> system: {}".format(n))
        logging.debug("--> input : {}".format(m))

        # next, we look for integrator chains
        logging.debug("Looking for integrator chains")

        # create symbolic variables to find integrator chains
        x_sym = ([sp.symbols('x%d' % k, type=float) for k in xrange(1,n+1)])
        u_sym = ([sp.symbols('u%d' % k, type=float) for k in xrange(1,m+1)])

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
        # where x_4 = d/dt x_3 and so on

        # find upper ends of integrator chains
        uppers = []
        for vv in chaindict.values():
            if (not chaindict.has_key(vv)):
                uppers.append(vv)

        # create ordered lists that temporarily represent the integrator chains
        tmpchains = []

        # therefore we flip the dictionary to work our way through its keys
        # (former values)
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
            logging.debug("--> found: {}".format(str(ic)))

        self.chains = chains
        self.x_sym = x_sym
        self.u_sym = u_sym

        # get minimal neccessary number of spline parts
        # for the manipulated variables
        # TODO: implement this!?
        # --> (3.35)      ?????
        #nu = -1
        #...
        #self.su = self.n - 3 + 2*(nu + 1)  ?????


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
                logging.error('Boundary values have to be strictly within the saturation limits!')
                logging.info('Please have a look at the documentation, \
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
        
        _ff_sym = sp.lambdify(self.x_sym+self.u_sym, ff, modules='sympy')
        
        def ff_sym(x,u):
            xu = np.hstack((x,u))
            return np.array(_ff_sym(*xu))
        
        # set altered boundary values and vectorfield function
        self.xa = xa
        self.xb = xb
        self.ff_sym = ff_sym
        
        # and backup the original ones, as well as some other stuff
        self.xa_orig = xa_orig
        self.xb_orig = xb_orig
        self.ff_sym_orig = ff_sym_orig
        
        self.x_sym_orig = x_sym_orig
        self.ff_orig = ff_num_orig
    
    
    def constrain(self):
        '''
        This method is used to determine the solution of the original constrained
        state variables by creating a composition of the saturation functions and
        the calculated solution for the introduced unconstrained variables.
        '''
        
        # get a copy of the current functions
        # (containing functions for unconstraint variables y_i)
        x_fnc = self.x_fnc.copy()
        dx_fnc = self.dx_fnc.copy()
        
        # iterate of all constraints
        for k, v in self.constraints.items():
            # get symbols of original constrained variable x_k, the introduced unconstrained variable y_k
            # the saturation limits y0, y1
            xk = self.x_sym_orig[k]
            yk = self.x_sym[k]
            y0, y1 = v
            
            # get the calculated solution function for the unconstrained variable and its derivative
            y_fnc = x_fnc[yk]
            dy_fnc = dx_fnc[yk]
            
            # create the compositions
            psi_y, dpsi_dy = saturation_functions(y_fnc, dy_fnc, y0, y1)
            
            # put created compositions into dictionaries of solution functions
            self.x_fnc[xk] = psi_y
            self.dx_fnc[xk] = dpsi_dy
            
            # remove solutions for unconstrained auxiliary variable and its derivative
            self.x_fnc.pop(yk)
            self.dx_fnc.pop(yk)
        
        # restore the original boundary values, variables and vectorfield function
        self.xa = self.xa_orig
        self.xb = self.xb_orig
        self.x_sym = self.x_sym_orig
        self.ff = self.ff_orig
    

    def startIteration(self):
        '''
        This is the main loop.

        At first the equations that have to be solved by collocation will be
        determined according to the integrator chains.

        Next, one step of the iteration is done by calling :meth:`iterate()`.

        After that, the accuracy of the found solution is checked.
        If it is within the tolerance range the iteration will stop.
        Else, the number of spline parts is raised and another step starts.
        
        
        Returns
        -------
        
        callable
            Callable function for the system state.
        
        callable
            Callable function for the input variables.
        '''

        logging.info("1st Iteration: {} spline parts".format(self.mparam['sx']))
        
        # resetting integrator chains according to value of self.use_chains
        if not self.mparam['use_chains']:
            self.chains = dict()

        # now we determine the equations that have to be solved by collocation
        # (--> lower ends of integrator chains)
        eqind = []

        if self.chains:
            # iterate over all integrator chains
            for ic in self.chains:
                # if lower end is a system variable
                # then its equation has to be solved
                if ic.lower.name.startswith('x'):
                    idx = self.x_sym.index(ic.lower)
                    eqind.append(idx)
            eqind.sort()
            
            # if every integrator chain ended with input variable
            if not eqind:
                eqind = range(self.n)
        else:
            # if integrator chains should not be used
            # then every equation has to be solved by collocation
            eqind = range(self.n)

        # save equation indices
        self.eqind = np.array(eqind)
        
        # start first iteration
        self.iterate()
        
        # check if desired accuracy is already reached
        self.checkAccuracy()
        
        # this was the first iteration
        # now we are getting into the loop
        while not self.reached_accuracy and self.nIt < self.mparam['maxIt']:
            # raise the number of spline parts
            self.mparam['sx'] = int(round(self.mparam['kx']*self.mparam['sx']))
            
            if self.nIt == 1:
                logging.info("2nd Iteration: {} spline parts".format(self.mparam['sx']))
            elif self.nIt == 2:
                logging.info("3rd Iteration: {} spline parts".format(self.mparam['sx']))
            elif self.nIt >= 3:
                logging.info("{}th Iteration: {} spline parts".format(self.nIt+1, self.mparam['sx']))

            # store the old spline to calculate the guess later
            self.old_splines = self.splines

            # start next iteration step
            self.iterate()

            # check if desired accuracy is reached
            self.checkAccuracy()
            
        # clear workspace
        self.clear()
        
        # as a last we, if there were any constraints to be taken care of,
        # we project the unconstrained variables back on the original
        # constrained ones
        if self.constraints:
            self.constrain()
        
        # return the found solution functions
        return self.x, self.u


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


    def iterate(self):
        '''
        This method is used to run one iteration step.

        First, new splines are initialised for the variables that are the upper
        end of an integrator chain.

        Then, a start value for the solver is determined and the equation
        system is build.

        Next, the equation system is solved and the resulting numerical values
        for the free parameters are written back.

        As a last, the initial value problem is simulated.
        '''
        self.nIt += 1
        
        # initialise splines
        with Timer("initSplines()"):
            self.initSplines()

        # Get first guess for solver
        with Timer("getGuess()"):
            self.getGuess()

        # create equation system
        with Timer("buildEQS()"):
            G, DG = self.buildEQS()

        # solve it
        with Timer("solveEQS()"):
            self.solveEQS(G, DG)

        # write back the coefficients
        with Timer("setCoeff()"):
            self.setCoeff()
        
        # solve the initial value problem
        with Timer("simulateIVP()"):
            self.simulateIVP()
    
    
    def getGuess(self):
        '''
        This method is used to determine a starting value (guess) for the
        solver of the collocation equation system.

        If it is the first iteration step, then a vector with the same length as
        the vector of the free parameters with arbitrarily values is returned.

        Else, for every variable a spline has been created for, the old spline
        of the iteration before and the new spline are evaluated at specific
        points and a equation system is solved which ensures that they are equal
        in these points.

        The solution of this system is the new start value for the solver.
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
                    logging.debug("Get new guess for spline %s".format(k.name))

                    # how many unknown coefficients does the new spline have
                    nn = len(self.indep_coeffs[k])

                    # and this will be the points to evaluate the old spline in
                    #   but we don't want to use the borders because they got
                    #   the boundary values already
                    #gpts = np.linspace(self.a,self.b,(nn+1),endpoint = False)[1:]
                    #gpts = np.linspace(self.a,self.b,(nn+1),endpoint = True)
                    gpts = np.linspace(self.a,self.b,nn,endpoint = True)

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

                    #TT = np.linalg.solve(NEW,OLD-NEW_abs)
                    TT = np.linalg.lstsq(NEW,OLD-NEW_abs)[0]
                    
                    guess = np.hstack((guess,TT))
                else:
                    # if it is a input variable, just take the old solution
                    guess = np.hstack((guess, self.coeffs_sol[k]))

        # the new guess
        self.guess = guess


    def initSplines(self):
        '''
        This method is used to initialise the spline objects.
        '''
        logging.debug("Initialise Splines")
        
        # dictionaries for splines and callable solution function for x, u and dx
        splines = dict()
        x_fnc = dict()
        u_fnc = dict()
        dx_fnc = dict()
        
        # make some stuff local
        sx = self.mparam['sx']
        su = self.mparam['su']

        # first handle variables that are part of an integrator chain
        for chain in self.chains:
            upper = chain.upper
            lower = chain.lower

            # here we just create a spline object for the upper ends of every chain
            # w.r.t. its lower end
            if lower.name.startswith('x'):
                splines[upper] = CubicSpline(self.a,self.b,n=sx,bc=[self.xa[upper],self.xb[upper]],steady=False,tag=upper.name)
                splines[upper].type = 'x'
            elif lower.name.startswith('u'):
                splines[upper] = CubicSpline(self.a,self.b,n=su,bc=[self.ua[lower],self.ub[lower]],steady=False,tag=upper.name)
                splines[upper].type = 'u'

            for i, elem in enumerate(chain.elements):
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
                        if splines[upper].type == 'u':
                            splines[upper].bcd = [self.ua[lower],self.ub[lower]]
                        x_fnc[elem] = splines[upper].f
                    if (i == 1):
                        splines[upper].bcd = [self.xa[elem],self.xb[elem]]
                        if splines[upper].type == 'u':
                            splines[upper].bcdd = [self.ua[lower],self.ub[lower]]
                        x_fnc[elem] = splines[upper].df
                    if (i == 2):
                        splines[upper].bcdd = [self.xa[elem],self.xb[elem]]
                        x_fnc[elem] = splines[upper].ddf

        # now handle the variables which are not part of any chain
        for xx in self.x_sym:
            if (not x_fnc.has_key(xx)):
                splines[xx] = CubicSpline(self.a,self.b,n=sx,bc=[self.xa[xx],self.xb[xx]],steady=False,tag=str(xx))
                splines[xx].type = 'x'
                x_fnc[xx] = splines[xx].f

        for i, uu in enumerate(self.u_sym):
            if (not u_fnc.has_key(uu)):
                splines[uu] = CubicSpline(self.a,self.b,n=su,bc=[self.ua[uu],self.ub[uu]],steady=False,tag=str(uu))
                splines[uu].type = 'u'
                u_fnc[uu] = splines[uu].f
        
        # solve smoothness conditions of each spline
        for ss in splines:
            with Timer("makesteady()"):
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
        This method is used to set up the equations for the collocation equation system
        and defines functions for the numerical evaluation of the system and its jacobian.
        
        Returns
        -------
        
        callable
            Function :py:func:`G(c)` that returns the collocation system 
            evaluated with numeric values for the independent parameters.
            
        callable
            Function :py:func:`DG(c)` that returns the jacobian matrix of the collocation system 
            w.r.t. the free parameters, evaluated with numeric values for them.
        
        '''

        logging.debug("Building Equation System")
        
        # make functions local
        x_fnc = self.x_fnc
        dx_fnc = self.dx_fnc
        u_fnc = self.u_fnc

        # make symbols local
        x_sym = self.x_sym
        u_sym = self.u_sym

        a = self.a
        b = self.b
        delta = self.mparam['delta']

        # now we generate the collocation points
        if self.mparam['colltype'] == 'equidistant':
            # get equidistant collocation points
            cpts = np.linspace(a,b,(self.mparam['sx']*delta+1),endpoint=True)
        elif self.mparam['colltype'] == 'chebychev':
            # determine rank of chebychev polynomial
            # of which to calculate zero points
            nc = int(self.mparam['sx']*delta - 1)

            # calculate zero points of chebychev polynomial --> in [-1,1]
            cheb_cpts = [np.cos( (2.0*i+1)/(2*(nc+1)) * np.pi) for i in xrange(nc)]
            cheb_cpts.sort()

            # transfer chebychev nodes from [-1,1] to our interval [a,b]
            a = self.a
            b = self.b
            chpts = [a + (b-a)/2.0 * (chp + 1) for chp in cheb_cpts]

            # add left and right borders
            cpts = np.hstack((a, chpts, b))
        else:
            logging.warning('Unknown type of collocation points.')
            logging.warning('--> will use equidistant points!')
            cpts = np.linspace(a,b,(self.mparam['sx']*delta+1),endpoint=True)

        lx = len(cpts)*len(x_sym)
        lu = len(cpts)*len(u_sym)

        Mx = [None]*lx
        Mx_abs = [None]*lx
        Mdx = [None]*lx
        Mdx_abs = [None]*lx
        Mu = [None]*lu
        Mu_abs = [None]*lu

        # here we do something that will be explained after we've done it  ;-)
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

        # as promised: here comes the explanation
        #
        # now, the dictionary 'indic' looks something like
        #
        # indic = {u1 : (0, 6), x3 : (18, 24), x4 : (24, 30), x1 : (6, 12), x2 : (12, 18)}
        #
        # which means, that in the vector of all independent parameters of all splines
        # the 0th up to the 5th item [remember: Python starts indexing at 0 and leaves out the last]
        # belong to the spline created for u1, the items with indices from 6 to 11 belong to the
        # spline created for x1 and so on...

        # total number of independent coefficients
        c_len = len(self.c_list)

        eqx = 0
        equ = 0
        for p in cpts:
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

        Mx = np.array(Mx)
        Mx_abs = np.array(Mx_abs)
        Mdx = np.array(Mdx)
        Mdx_abs = np.array(Mdx_abs)
        Mu = np.array(Mu)
        Mu_abs = np.array(Mu_abs)
        
        # here we create a callable function for the jacobian matrix of the vectorfield
        # w.r.t. to the system and input variables
        f = self.ff_sym(x_sym,u_sym)
        Df_mat = sp.Matrix(f).jacobian(x_sym+u_sym)
        Df = sp.lambdify(x_sym+u_sym, Df_mat, modules='numpy')

        # the following would be created with every call to self.DG but it is possible to
        # only do it once. So we do it here to speed things up.

        # here we compute the jacobian matrix of the derivatives of the system state functions
        # (as they depend on the free parameters in a linear fashion its just the above matrix Mdx)
        DdX = Mdx.reshape((len(cpts),-1,len(self.c_list)))[:,self.eqind,:]
        DdX = np.vstack(DdX)

        # here we compute the jacobian matrix of the system/input functions as they also depend on
        # the free parameters
        DXU = []
        x_len = len(self.x_sym)
        u_len = len(self.u_sym)
        xu_len = x_len + u_len

        for i in xrange(len(cpts)):
            DXU.append(np.vstack(( Mx[x_len*i:x_len*(i+1)], Mu[u_len*i:u_len*(i+1)] )))
        DXU_old = DXU
        DXU = np.vstack(DXU)
        
        if self.mparam['use_sparse']:
            Mx = sparse.csr_matrix(Mx)
            Mx_abs = sparse.csr_matrix(Mx_abs)
            Mdx = sparse.csr_matrix(Mdx)
            Mdx_abs = sparse.csr_matrix(Mdx_abs)
            Mu = sparse.csr_matrix(Mu)
            Mu_abs = sparse.csr_matrix(Mu_abs)

            DdX = sparse.csr_matrix(DdX)
            DXU = sparse.csr_matrix(DXU)
        
        # now we are going to create a callable function for the equation system
        # and its jacobian
            
        # make some stuff local
        ff = self.ff
        
        eqind = self.eqind
        
        cp_len = len(cpts)
        
        # templates
        F = np.empty((cp_len, len(eqind)))
        DF = sparse.dok_matrix( (cp_len*x_len, xu_len*cp_len) )
        
        # define the callable functions for the eqs
        def G(c):
            X = Mx.dot(c) + Mx_abs
            U = Mu.dot(c) + Mu_abs

            X = np.array(X).reshape((-1,x_len))
            U = np.array(U).reshape((-1,u_len))

            # evaluate system equations and select those related
            # to lower ends of integrator chains (via eqind)
            # other equations need not to be solved
            #F = np.empty((cp_len, len(eqind)))
            
            for i in xrange(cp_len):
                F[i,:] = ff(X[i], U[i])[eqind]
            
            dX = Mdx.dot(c) + Mdx_abs
            dX = np.array(dX).reshape((-1,x_len))[:,eqind]
        
            G = F - dX
            
            return G.ravel()
        
        # and its jacobian
        def DG(c):
            '''
            Returns the jacobian matrix of the collocation system w.r.t. the
            independent parameters evaluated at :attr:`c`.
            '''
            
            # first we calculate the x and u values in all collocation points
            # with the current numerical values of the free parameters
            X = Mx.dot(c) + Mx_abs
            X = np.array(X).reshape((-1,x_len)) # one column for every state component
            U = Mu.dot(c) + Mu_abs
            U = np.array(U).reshape((-1,u_len)) # one column for every input component
            
            # now we iterate over every row
            # (one for every collocation point)
            for i in xrange(X.shape[0]):
                # concatenate the values so that they can be unpacked at once
                tmp_xu = np.hstack((X[i], U[i]))
                
                # calculate one block of the jacobian
                DF_block = Df(*tmp_xu).tolist()
                
                # the sparse format (dok) only allows us to add multiple values
                # in one dimension, so we cannot add the block at once but
                # have to iterate over its rows
                for j in xrange(x_len):
                    DF[x_len*i+j, xu_len*i:xu_len*(i+1)] = DF_block[j]
            
            # now transform it into another sparse format that is more suitable
            # for calculating matrix products,
            # and then do so with `DXU`
            DF_csr = sparse.csr_matrix(DF).dot(DXU)
            
            # TODO: explain the following code
            if self.mparam['use_chains']:
                DF_csr = [row for idx,row in enumerate(DF_csr.toarray()[:]) if idx%x_len in eqind]
                DF_csr = sparse.csr_matrix(DF_csr)
            
            DG = DF_csr - DdX
            
            return DG
        
        # return the callable functions
        return G, DG


    def solveEQS(self, G, DG):
        '''
        This method is used to solve the collocation equation system.
        
        Parameters
        ----------
        
        G : callable
            Function that "evaluates" the equation system.
        
        DG : callable
            Function for the jacobian.
        '''

        logging.debug("Solving Equation System")
        
        # create our solver
        solver = Solver(F=G, DF=DG, x0=self.guess, tol=self.mparam['tol'],
                        maxIt=self.mparam['sol_steps'], method=self.mparam['method'])
        
        # solve the equation system
        self.sol = solver.solve()
        
    
    def setCoeff(self):
        '''
        Set found numerical values for the independent parameters of each spline.

        This method is used to get the actual splines by using the numerical
        solutions to set up the coefficients of the polynomial spline parts of
        every created spline.
        '''

        logging.debug("Set spline coefficients")

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


    def simulateIVP(self):
        '''
        This method is used to solve the initial value problem.
        '''

        logging.debug("Solving Initial Value Problem")
        
        # calulate simulation time
        T = self.b - self.a
        
        # get list of start values
        start = []
        
        if self.constraints:
            for xx in self.x_sym_orig:
                start.append(self.xa_orig[xx])
            
            # create simulation object
            S = Simulation(self.ff_orig, T, start, self.u)
        else:
            for xx in self.x_sym:
                start.append(self.xa[xx])
            
            # create simulation object
            S = Simulation(self.ff, T, start, self.u)
        
        logging.debug("start: {}".format(str(start)))
        
        # start forward simulation
        self.sim = S.simulate()


    def checkAccuracy(self):
        '''
        Checks whether the desired accuracy for the boundary values was reached.

        It calculates the difference between the solution of the simulation
        and the given boundary values at the right border and compares its
        maximum against the tolerance set by self.eps
        '''
        
        
        # this is the solution of the simulation
        xt = self.sim[1]

        # what is the error
        logging.debug(40*"-")
        logging.debug("Ending up with:   Should Be:  Difference:")

        err = np.empty(self.n)
        if self.constraints:
            for i, xx in enumerate(self.x_sym_orig):
                err[i] = abs(self.xb_orig[xx] - xt[-1][i])
                logging.debug(str(xx)+" : %f     %f    %f"%(xt[-1][i], self.xb_orig[xx], err[i]))
        else:
            for i, xx in enumerate(self.x_sym):
                err[i] = abs(self.xb[xx] - xt[-1][i])
                logging.debug(str(xx)+" : %f     %f    %f"%(xt[-1][i], self.xb[xx], err[i]))
        
        logging.debug(40*"-")
        
        if self.mparam['ierr']:
            # calculate maximum consistency error on the whole interval
            maxH = consistency_error((self.a,self.b), self.x, self.u, self.dx, self.ff)
            
            self.reached_accuracy = (maxH < self.mparam['ierr']) and (max(err) < self.mparam['eps'])
            logging.debug('maxH = %f'%maxH)
        else:
            # just check if tolerance for the boundary values is satisfied
            self.reached_accuracy = max(err) < self.mparam['eps']
        
        if self.reached_accuracy or self.nIt == self.mparam['maxIt']:
            logging.info("  --> reached desired accuracy: "+str(self.reached_accuracy))
        else:
            logging.debug("  --> reached desired accuracy: "+str(self.reached_accuracy))
    

    def x(self, t):
        '''
        Returns the current system state.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the system at.
        '''
        
        if not self.a <= t <= self.b:
            logging.warning("Time point 't' has to be in (a,b)")
            arr = None
        else:
            arr = np.array([self.x_fnc[xx](t) for xx in self.x_sym])
        
        return arr


    def u(self, t):
        '''
        Returns the state of the input variables.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the input variables at.
        '''
        
        if not self.a <= t <= self.b+0.05:
            #logging.warning("Time point 't' has to be in (a,b)")
            arr = None
            arr = np.array([self.u_fnc[uu](self.b) for uu in self.u_sym])
        else:
            arr = np.array([self.u_fnc[uu](t) for uu in self.u_sym])
        
        return arr


    def dx(self, t):
        '''
        Returns the state of the 1st derivatives of the system variables.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the 1st derivatives at.
        '''
        
        if not self.a <= t <= self.b+0.05:
            logging.warning("Time point 't' has to be in (a,b)")
            arr = None
        else:
            arr = np.array([self.dx_fnc[xx](t) for xx in self.x_sym])
        
        return arr


    def plot(self):
        '''
        Plot the calculated trajectories and error functions.

        This method calculates the error functions and then calls
        the :func:`utilities.plotsim` function.
        '''

        try:
            import matplotlib
        except ImportError:
            logging.error('Matplotlib is not available for plotting.')
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

