
# IMPORTS
import numpy as np
import sympy as sp
from scipy import sparse
import pickle

from trajectories import Trajectory
from collocation import CollocationSystem
from simulation import Simulator
import auxiliary
import visualisation
from log import logging, Timer

from IPython import embed as IPS


class ControlSystem(object):
    '''
    Base class of the PyTrajectory project.

    Parameters
    ----------

    ff :  callable
        Vector field (rhs) of the control system.
    
    a : float
        Left border of the considered time interval.
    
    b : float
        Right border of the considered time interval.
    
    xa : list
        Boundary values at the left border.
    
    xb : list
        Boundary values at the right border.
    
    ua : list
        Boundary values of the input variables at left border.
    
    ub : list
        Boundary values of the input variables at right border.
    
    constraints : dict
        Box-constraints of the state variables.
    
    kwargs
        ============= =============   ============================================================
        key           default value   meaning
        ============= =============   ============================================================
        sx            5               Initial number of spline parts for the system variables
        su            5               Initial number of spline parts for the input variables
        kx            2               Factor for raising the number of spline parts
        delta         2               Constant for calculation of collocation points
        maxIt         10              Maximum number of iteration steps
        eps           1e-2            Tolerance for the solution of the initial value problem
        ierr          1e-1            Tolerance for the error on the whole interval
        tol           1e-5            Tolerance for the solver of the equation system
        method        'leven'         The solver algorithm to use
        use_chains    True            Whether or not to use integrator chains
        coll_type     'equidistant'   The type of the collocation points
        sol_steps     100             Maximum number of iteration steps for the eqs solver
        nodes_type    'equidistant'   The type of the spline nodes
        ============= =============   ============================================================
    
    '''
    def __init__(self, ff, a=0.0, b=1.0, xa=[], xb=[], ua=[], ub=[], constraints=None, **kwargs):
        # Save the symbolic vectorfield
        self.ff_sym = ff
        
        # The borders of the considered time interval
        self.a = a
        self.b = b
        
        # Set default values for method parameters
        method_param = {'sx' : 5,
                        'su' : 5,
                        'kx' : 2,
                        'delta' : 2,
                        'maxIt' : 10,
                        'eps' : 1e-2,
                        'ierr' : 1e-1,
                        'tol' : 1e-5,
                        'method' : 'leven',
                        'use_chains' : True,
                        'coll_type' : 'equidistant',
                        'sol_steps' : 100,
                        'nodes_type' : 'equidistant'}
        
        # Change default values of given kwargs
        for k, v in kwargs.items():
            if method_param.has_key(k):
                method_param[k] = v
        
        # Analyse the given system to set some parameters
        n, m, x_sym, u_sym, chains, eqind = self.analyse(xa)
        
        # a little check
        if not (len(xa) == len(xb) == n):
            raise ValueError('Dimension mismatch xa, xb')
        
        # Set system dimensions
        self.n = n
        self.m = m
        
        # Set symbols for state and input variables
        self.x_sym = [sym.name for sym in x_sym]
        self.u_sym = [sym.name for sym in u_sym]
        
        # Set integrator chains and equations that have to be solved
        self.chains = chains
        self.eqind = eqind
        
        # Transform lists of boundary values into dictionaries
        boundary_values = dict()
        
        for i, xx in enumerate(self.x_sym):
            boundary_values[xx] = (xa[i], xb[i])
        
        if ua and ub:
            for i, uu in enumerate(self.u_sym):
                boundary_values[uu] = (ua[i], ub[i])
        elif ua and not ub:
            for i, uu in enumerate(self.u_sym):
                boundary_values[uu] = (ua[i], None)
        elif not ua and ub:
            for i, uu in enumerate(self.u_sym):
                boundary_values[uu] = (None, ub[i])
        elif not ua and not ub:
            for i, uu in enumerate(self.u_sym):
                boundary_values[uu] = (None, None)
        
        self._boundary_values = boundary_values
        
        # Handle system constraints if there are any
        self.constraints = constraints
        if self.constraints:
            # transform the constrained vectorfield into an unconstrained one
            ff_sym, boundary_values, orig_backup = self.unconstrain()
            
            self.ff_sym = ff_sym
            self._boundary_values = boundary_values
            self.orig_backup = orig_backup
            
            # we cannot make use of an integrator chain
            # if it contains a constrained variable
            method_param['use_chains'] = False
            # TODO: implement it so that just those chains are not used 
            #       which actually contain a constrained variable
        
        # Now we transform the symbolic function of the vector field to
        # a numeric one for faster evaluation
        self.ff = auxiliary.sym2num_vectorfield(self.ff_sym, self.x_sym, self.u_sym, vectorized=False)
        
        # Create trajectory object
        self.trajectories = Trajectory(sys=self, sx=method_param['sx'], su=method_param['su'],
                                        use_chains=method_param['use_chains'],
                                        nodes_type=method_param['nodes_type'])

        # and equation system objects
        self.eqs = CollocationSystem(sys=self, tol=method_param['tol'], sol_steps=method_param['sol_steps'], 
                                        method=method_param['method'], coll_type=method_param['coll_type'])
        
        # Reset iteration number
        self.nIt = 0
        self._maxIt = method_param['maxIt']
        self._kx = method_param['kx']
        
        # We didn't really do anything yet, so this should be false
        self.reached_accuracy = False
        self._ierr = method_param['ierr']
        self._eps = method_param['eps']

        # and likewise this should not be existent yet
        self.sol = None
    
    
    def analyse(self, xa):
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
            Inputs dimension.
        
        list
            List of sympy.symbols for state variables
        
        list
            List of sympy.symbols for input variables
        
        list
            Found integrator chains of the system.
        
        list
            Indices of the equations that have to be solved using collocation.
        '''

        logging.info("Analysing System Structure")
        
        # first, determine system dimensions
        logging.debug("Determine system/input dimensions")
        
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
        
        logging.debug("--> system: %d"%n)
        logging.debug("--> input : %d"%m)

        # next, we look for integrator chains
        logging.debug("Looking for integrator chains")

        # create symbolic variables to find integrator chains
        x_sym = ([sp.symbols('x%d' % k, type=float) for k in xrange(1,n+1)])
        u_sym = ([sp.symbols('u%d' % k, type=float) for k in xrange(1,m+1)])

        fi = self.ff_sym(x_sym, u_sym)

        chains, eqind = auxiliary.find_integrator_chains(fi, x_sym, u_sym)
        
        # if we don't take advantage of the system structure
        # we need to solve every equation
        #if not self.mparam['use_chains']:
        #    eqind = range(len(x_sym))
        
        # get minimal neccessary number of spline parts
        # for the manipulated variables
        # TODO: implement this!?
        # --> (3.35)      ?????
        #nu = -1
        #...
        #self.su = self.n - 3 + 2*(nu + 1)  ?????
        
        return n, m, x_sym, u_sym, chains, eqind
    
    def set_param(self, param='', value=None):
        '''
        Alters the value of the method parameters.
        
        Parameters
        ----------
        
        param : str
            The method parameter to alter
        
        value
            The new value
        '''
        
        if param in {'kx', 'maxIt', 'eps', 'ierr'}:
            setattr(self, '_{}'.format(param), value)
        elif param in {'sx', 'su', 'use_chains', 'nodes_type'}:
            if param == 'nodes_type' and value != 'equidistant':
                raise NotImplementedError()
            setattr(self.trajectories, '_{}'.format(param), value)
        elif param in {'tol', 'method', 'coll_type', 'sol_steps'}:
            setattr(self.eqs, '_{}'.format(param), value)
        else:
            raise AttributeError("Invalid method parameter ({})".format(param))
    
    def unconstrain(self):
        '''
        This method is used to enable compliance with desired box constraints given by the user.
        It transforms the vectorfield by projecting the constrained state variables on
        new unconstrained ones.
        '''
        
        # make some stuff local
        ff = sp.Matrix(self.ff_sym(sp.symbols(self.x_sym), sp.symbols(self.u_sym)))
        boundary_values = self._boundary_values
        x_sym = self.x_sym
        
        # First, we backup all things that will be influenced in some way
        #
        # backup original state variables and their boundary values
        x_sym_orig = 1*x_sym
        boundary_values_orig = boundary_values.copy()
        
        # backup symbolic vectorfield function
        ff_sym_orig = self.ff_sym
        
        # create a numeric vectorfield function of the original vectorfield
        # and back it up (will be used in simulation step of the main iteration)
        ff_num_orig = auxiliary.sym2num_vectorfield(ff_sym_orig, x_sym_orig, self.u_sym)
        
        # Now we can handle the constraints by projecting the constrained state variables
        # on new unconstrained variables using saturation functions
        for k, v in self.constraints.items():
            # check if boundary values are within saturation limits
            bv_a, bv_b = boundary_values[x_sym[k]]
            if not ( v[0] < bv_a < v[1] ) or \
                not ( v[0] < bv_b < v[1] ):
                logging.error('Boundary values have to be strictly within the saturation limits!')
                logging.info('Please have a look at the documentation, \
                              especially the example of the constrained double intgrator.')
                raise ValueError('Boundary values have to be strictly within the saturation limits!')
            
            # replace constrained state variable with new unconstrained one
            x_sym[k] = 'y{}'.format(k+1)
            
            # calculate saturation function expression and its derivative
            yk = x_sym[k]
            yk_sym = sp.Symbol(yk)
            m = 4.0/(v[1] - v[0])
            psi = v[1] - (v[1]-v[0])/(1.0+sp.exp(m*yk_sym))
            
            #dpsi = ((v[1]-v[0])*m*sp.exp(m*yk))/(1.0+sp.exp(m*yk))**2
            dpsi = (4.0*sp.exp(m*yk_sym))/(1.0+sp.exp(m*yk_sym))**2
            
            # replace constrained variables in vectorfield with saturation expression
            # x(t) = psi(y(t))
            ff = ff.replace(sp.Symbol(x_sym_orig[k]), psi)
            
            # update vectorfield to represent differential equation for new
            # unconstrained state variable
            #
            # d/dt x(t) = (d/dy psi(y(t))) * d/dt y(t)
            # <==> d/dt y(t) = d/dt x(t) / (d/dy psi(y(t)))
            ff[k] = ff[k] / dpsi
            
            # replace key of constrained variable in dictionaries for boundary values
            # with new symbol for the unconstrained variable
            xk = x_sym_orig[k]
            boundary_values[yk] = boundary_values.pop(xk)
            
            # update boundary values for new unconstrained variable
            wa, wb = boundary_values[yk]
            boundary_values[yk] = ( (1.0/m)*np.log((wa-v[0])/(v[1]-wa)),
                                    (1.0/m)*np.log((wb-v[0])/(v[1]-wb)) )
        
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
        orig_backup = {'boundary_values' : boundary_values_orig,
                        'ff_sym' : ff_sym_orig, 'ff_num' : ff_num_orig,
                        'x_sym' : x_sym_orig}
        
        return ff_sym, boundary_values, orig_backup
    
    def constrain(self):
        '''
        This method is used to determine the solution of the original constrained
        state variables by creating a composition of the saturation functions and
        the calculated solution for the introduced unconstrained variables.
        '''
        
        # get a copy of the current function dictionaries
        # (containing functions for unconstrained variables y_i)
        x_fnc = self.trajectories._x_fnc.copy()
        dx_fnc = self.trajectories._dx_fnc.copy()
        
        # iterate over all constraints
        for k, v in self.constraints.items():
            # get symbols of original constrained variable x_k, the introduced unconstrained variable y_k
            # and the saturation limits y0, y1
            xk = self.orig_backup['x_sym'][k]
            yk = self.x_sym[k]
            y0, y1 = v
            
            # get the calculated solution function for the unconstrained variable and its derivative
            y_fnc = x_fnc[yk]
            dy_fnc = dx_fnc[yk]
            
            # create the compositions
            psi_y, dpsi_dy = auxiliary.saturation_functions(y_fnc, dy_fnc, y0, y1)
            
            # put created compositions into dictionaries of solution functions
            self.trajectories._x_fnc[xk] = psi_y
            self.trajectories._dx_fnc[xk] = dpsi_dy
            
            # remove solutions for unconstrained auxiliary variable and its derivative
            self.trajectories._x_fnc.pop(yk)
            self.trajectories._dx_fnc.pop(yk)
        
        # restore the original boundary values, variables and vectorfield functions
        # TODO: should the constrained stuff be saved (not longer needed?)
        self._boundary_values = self.orig_backup['boundary_values']
        self.x_sym = self.orig_backup['x_sym']
        self.ff = self.orig_backup['ff_num']
        self.ff_sym = self.orig_backup['ff_sym']
        self.trajectories._x_sym = self.x_sym
    
    def solve(self):
        '''
        This is the main loop.
        
        While the desired accuracy has not been reached, the number of
        spline parts is raised and one iteration step is done.
        
        Returns
        -------
        
        callable
            Callable function for the system state.
        
        callable
            Callable function for the input variables.
        '''
        
        # do the first iteration step
        logging.info("1st Iteration: %d spline parts"%self.trajectories._sx)
        self._iterate()
        
        # this was the first iteration
        # now we are getting into the loop
        while not self.reached_accuracy and self.nIt < self._maxIt:
            # raise the number of spline parts
            self.trajectories._sx = int(round(self._kx * self.trajectories._sx))
            
            if self.nIt == 1:
                logging.info("2nd Iteration: {} spline parts".format(self.trajectories._sx))
            elif self.nIt == 2:
                logging.info("3rd Iteration: {} spline parts".format(self.trajectories._sx))
            elif self.nIt >= 3:
                logging.info("{}th Iteration: {} spline parts".format(self.nIt+1, self.trajectories._sx))

            # start next iteration step
            self._iterate()

        # as a last we, if there were any constraints to be taken care of,
        # we project the unconstrained variables back on the original constrained ones
        if self.constraints:
            self.constrain()
        
        # return the found solution functions
        return self.trajectories.x, self.trajectories.u
    
    def _iterate(self):
        '''
        This method is used to run one iteration step.

        First, new splines are initialised for the variables that are the upper
        end of an integrator chain.

        Then, a start value for the solver is determined and the equation
        system is set up.

        Next, the equation system is solved and the resulting numerical values
        for the free parameters are applied to the corresponding splines.

        As a last, the resulting initial value problem is simulated.
        '''
        
        # Increase iteration number
        self.nIt += 1
        
        # Initialise the spline function objects
        self.trajectories.init_splines(boundary_values=self._boundary_values)
        
        # Get an initial value (guess)
        self.eqs.get_guess(trajectories=self.trajectories)
        
        # Build the collocation equations system
        with Timer('Building equation system'):
            G, DG = self.eqs.build(sys=self, trajectories=self.trajectories)
        
        # Solve the collocation equation system
        sol = self.eqs.solve(G, DG)
        
        # Set the found solution
        self.trajectories.set_coeffs(sol)
        
        # Solve the resulting initial value problem
        self.simulate()
        
        # check if desired accuracy is reached
        self.check_accuracy()
    
    def simulate(self):
        '''
        This method is used to solve the resulting initial value problem
        after the computation of a solution for the input trajectories.
        '''

        logging.debug("Solving Initial Value Problem")
        
        # calulate simulation time
        T = self.b - self.a
        
        # get list of start values
        start = []
        
        if self.constraints:
            x_vars = self.orig_backup['x_sym']
            start_dict = dict([(k, v[0]) for k, v in self.orig_backup['boundary_values'].items() if k in x_vars])
            ff = self.orig_backup['ff_num']
        else:
            x_vars = self.x_sym
            start_dict = dict([(k, v[0]) for k, v in self._boundary_values.items() if k in x_vars])
            ff = self.ff
        
        for x in x_vars:
            start.append(start_dict[x])
        
        # create simulation object
        S = Simulator(ff, T, start, self.trajectories.u)
        
        logging.debug("start: %s"%str(start))
        
        # start forward simulation
        self.sim_data = S.simulate()
    
    def check_accuracy(self):
        '''
        Checks whether the desired accuracy for the boundary values was reached.

        It calculates the difference between the solution of the simulation
        and the given boundary values at the right border and compares its
        maximum against the tolerance.
        
        If set by the user it also calculates some kind of consistency error
        that shows how "well" the spline functions comply with the system
        dynamic given by the vector field.
        
        '''
        
        # this is the solution of the simulation
        a = self.sim_data[0][0]
        b = self.sim_data[0][-1]
        xt = self.sim_data[1]
        
        # get boundary values at right border of the interval
        if self.constraints:
            bv = self.orig_backup['boundary_values']
            x_sym = self.orig_backup['x_sym'] 
        else:
            bv = self._boundary_values
            x_sym = self.x_sym
            
        xb = dict([(k, v[1]) for k, v in bv.items() if k in x_sym])
        
        # what is the error
        logging.debug(40*"-")
        logging.debug("Ending up with:   Should Be:  Difference:")

        err = np.empty(xt.shape[1])
        for i, xx in enumerate(x_sym):
            err[i] = abs(xb[xx] - xt[-1][i])
            logging.debug(str(xx)+" : %f     %f    %f"%(xt[-1][i], xb[xx], err[i]))
        
        logging.debug(40*"-")
        
        if self._ierr:
            # calculate maximum consistency error on the whole interval
            maxH = auxiliary.consistency_error((a,b), self.trajectories.x, self.trajectories.u, self.trajectories.dx, self.ff)
            
            reached_accuracy = (maxH < self._ierr) and (max(err) < self._eps)
            logging.debug('maxH = %f'%maxH)
        else:
            # just check if tolerance for the boundary values is satisfied
            reached_accuracy = max(err) < self._eps
        
        if reached_accuracy:
            logging.info("  --> reached desired accuracy: "+str(reached_accuracy))
        else:
            logging.debug("  --> reached desired accuracy: "+str(reached_accuracy))
        
        self.reached_accuracy = reached_accuracy
    
    def plot(self):
        '''
        Plot the calculated trajectories and show interval error functions.

        This method calculates the error functions and then calls
        the :py:func:`visualisation.plotsim` function.
        '''

        try:
            import matplotlib
        except ImportError:
            logging.error('Matplotlib is not available for plotting.')
            return

        # calculate the error functions H_i(t)
        max_con_err, error = auxiliary.consistency_error((self.a, self.b), 
                                                          self.trajectories.x,
                                                          self.trajectories.u, 
                                                          self.trajectories.dx, 
                                                          self.ff, len(self.sim_data[0]), True)
        
        H = dict()
        for i in self.eqind:
            H[i] = error[:,i]

        # call utilities.plotsim()
        visualisation.plot_simulation(self.sim_data, H)
        # t = self.sim_data[0]
        # xt = np.array([self.trajectories.x(tt) for tt in t])
        # ut = self.sim_data[2]
        # visualisation.plot_simulation([t,xt,ut], H)
    
    def save(self, fname=''):
        '''
        Save data using the module :py:mod:`pickle`.
        
        Currently only saves simulation data.
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
        save['sim_data'] = self.sim_data
        
        if not fname:
            fname = __file__.split('.')[0] + '.pcl'
        elif not fname.endswith('.pcl'):
            fname += '.pcl'
        
        with open(fname, 'wb') as dumpfile:
            pickle.dump(save, dumpfile)
        
