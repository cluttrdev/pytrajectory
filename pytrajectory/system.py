
# IMPORTS
import numpy as np
import sympy as sp
from scipy import sparse
import pickle
import copy

from trajectories import Trajectory
from collocation import CollocationSystem
from simulation import Simulator
import auxiliary
import visualisation
from log import logging, Timer

# DEBUGGING
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

    def __init__(self, ff, a=0., b=1., xa=[], xb=[], ua=[], ub=[], constraints=None, **kwargs):
        # set method parameters
        self._parameters = dict()
        self._parameters['maxIt'] = kwargs.get('maxIt', 10)
        self._parameters['eps'] = kwargs.get('eps', 1e-2)
        self._parameters['ierr'] = kwargs.get('ierr', 1e-1)

        # create an object for the dynamical system
        self.dyn_sys = DynamicalSystem(f_sym=ff, a=a, b=b, xa=xa, xb=xb, ua=ua, ub=ub)

        # handle eventual system constraints
        self.constraints = constraints
        if self.constraints is not None:
            # transform the constrained vectorfield into an unconstrained one
            self.unconstrain(constraints)

            # we cannot make use of an integrator chain
            # if it contains a constrained variable
            kwargs['use_chains'] = False
            # TODO: implement it so that just those chains are not used 
            #       which actually contain a constrained variable

        # create an object for the collocation equation system
        self.eqs = CollocationSystem(sys=self.dyn_sys, **kwargs)

        # We didn't really do anything yet, so this should be false
        self.reached_accuracy = False

    def set_param(self, param='', value=None):
        '''
        Alters the value of the method parameters.
        
        Parameters
        ----------
        
        param : str
            The method parameter
        
        value
            The new value
        '''
        
        if param in {'maxIt', 'eps', 'ierr'}:
            self._parameters[param] = value

        elif param in {'n_parts_x', 'sx', 'n_parts_u', 'su', 'kx', 'use_chains', 'nodes_type', 'use_std_approach'}:
            if param == 'nodes_type' and value != 'equidistant':
                raise NotImplementedError()

            if param == 'sx':
                param = 'n_parts_x'
            if param == 'su':
                param = 'n_parts_u'

            self.eqs.trajectories._parameters[param] = value

        elif param in {'tol', 'method', 'coll_type', 'sol_steps'}:
            self.eqs._parameters[param] = value

        else:
            raise AttributeError("Invalid method parameter ({})".format(param))
        
    def unconstrain(self, constraints):
        '''
        This method is used to enable compliance with desired box constraints given by the user.
        It transforms the vectorfield by projecting the constrained state variables on
        new unconstrained ones.

        Parameters
        ----------

        constraints : dict
            The box constraints for the state variables
        '''

        # save constraints
        self.constraints = constraints

        # backup the original constrained system
        self._dyn_sys_orig = copy.deepcopy(self.dyn_sys)

        # get symbolic vectorfield
        # (as sympy matrix toenable replacement method)
        x = sp.symbols(self.dyn_sys.states)
        u = sp.symbols(self.dyn_sys.inputs)
        ff_mat = sp.Matrix(self.dyn_sys.f_sym(x, u))

        # get neccessary information form the dynamical system
        a = self.dyn_sys.a
        b = self.dyn_sys.b
        boundary_values = self.dyn_sys.boundary_values
        
        # handle the constraints by projecting the constrained state variables
        # on new unconstrained variables using saturation functions
        for k, v in self.constraints.items():
            # check if boundary values are within saturation limits
            xk = self.dyn_sys.states[k]
            xa, xb = self.dyn_sys.boundary_values[xk]
            
            if not ( v[0] < xa < v[1] ) or not ( v[0] < xb < v[1] ):
                logging.error('Boundary values have to be strictly within the saturation limits!')
                logging.info('Please have a look at the documentation, \
                              especially the example of the constrained double intgrator.')
                raise ValueError('Boundary values have to be strictly within the saturation limits!')
            
            # calculate saturation function expression and its derivative
            yk = sp.Symbol(xk)
            m = 4.0/(v[1] - v[0])
            psi = v[1] - (v[1]-v[0])/(1. + sp.exp(m * yk))
            
            #dpsi = ((v[1]-v[0])*m*sp.exp(m*yk))/(1.0+sp.exp(m*yk))**2
            dpsi = (4. * sp.exp(m * yk))/(1. + sp.exp(m * yk))**2
            
            # replace constrained variables in vectorfield with saturation expression
            # x(t) = psi(y(t))
            ff_mat = ff_mat.replace(sp.Symbol(xk), psi)
            
            # update vectorfield to represent differential equation for new
            # unconstrained state variable
            #
            #      d/dt x(t) = (d/dy psi(y(t))) * d/dt y(t)
            # <==> d/dt y(t) = d/dt x(t) / (d/dy psi(y(t)))
            ff_mat[k] /= dpsi
            
            # update boundary values for new unconstrained variable
            boundary_values[xk] = ( (1./m) * np.log((xa - v[0]) / (v[1] - xa)),
                                    (1./m) * np.log((xb - v[0]) / (v[1] - xb)) )
        
        # create a callable function for the new symbolic vectorfield
        ff = np.asarray(ff_mat).flatten().tolist()
        xu = self.dyn_sys.states + self.dyn_sys.inputs
        _f_sym = sp.lambdify(xu, ff, modules='sympy')
        def f_sym(x, u):
            xu = np.hstack((x,u))
            return _f_sym(*xu)

        # create a new unconstrained system
        xa = [boundary_values[x][0] for x in self.dyn_sys.states]
        xb = [boundary_values[x][1] for x in self.dyn_sys.states]
        ua = [boundary_values[u][0] for u in self.dyn_sys.inputs]
        ub = [boundary_values[u][1] for u in self.dyn_sys.inputs]

        self.dyn_sys = DynamicalSystem(f_sym , a, b, xa, xb, ua, ub)

    def constrain(self):
        '''
        This method is used to determine the solution of the original constrained
        state variables by creating a composition of the saturation functions and
        the calculated solution for the introduced unconstrained variables.
        '''
        
        # get a copy of the current function dictionaries
        # (containing functions for unconstrained variables y_i)
        x_fnc = copy.deepcopy(self.eqs.trajectories.x_fnc)
        dx_fnc = copy.deepcopy(self.eqs.trajectories.dx_fnc)
        
        # iterate over all constraints
        for k, v in self.constraints.items():
            # get symbols of original constrained variable x_k, the introduced unconstrained variable y_k
            # and the saturation limits y0, y1
            xk = self._dyn_sys_orig.states[k]
            yk = self.dyn_sys.states[k]
            y0, y1 = v
            
            # get the calculated solution function for the unconstrained variable and its derivative
            y_fnc = x_fnc[yk]
            dy_fnc = dx_fnc[yk]
            
            # create the compositions
            psi_y, dpsi_dy = auxiliary.saturation_functions(y_fnc, dy_fnc, y0, y1)
            
            # put created compositions into dictionaries of solution functions
            self.eqs.trajectories.x_fnc[xk] = psi_y
            self.eqs.trajectories.dx_fnc[xk] = dpsi_dy
            
            # remove solutions for unconstrained auxiliary variable and its derivative
            #self.eqs.trajectories.x_fnc.pop(yk)
            #self.eqs.trajectories.dx_fnc.pop(yk)
        
        # restore the original boundary values, variables and vectorfield functions
        # TODO: should the constrained stuff be saved (not longer needed?)
        #self._boundary_values = self.orig_backup['boundary_values']
        #self.x_sym = self.orig_backup['x_sym']
        #self.ff = self.orig_backup['ff_num']
        #self.ff_sym = self.orig_backup['ff_sym']
        #self.eqs.trajectories._x_sym = self.x_sym
        
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
        logging.info("1st Iteration: {} spline parts".format(self.eqs.trajectories.n_parts_x))
        self._iterate()
        
        # this was the first iteration
        # now we are getting into the loop
        self.nIt = 1
        
        while not self.reached_accuracy and self.nIt < self._parameters['maxIt']:
            # raise the number of spline parts
            self.eqs.trajectories._raise_spline_parts()
            
            if self.nIt == 1:
                logging.info("2nd Iteration: {} spline parts".format(self.eqs.trajectories.n_parts_x))
            elif self.nIt == 2:
                logging.info("3rd Iteration: {} spline parts".format(self.eqs.trajectories.n_parts_x))
            elif self.nIt >= 3:
                logging.info("{}th Iteration: {} spline parts".format(self.nIt+1, self.eqs.trajectories.n_parts_x))

            # start next iteration step
            self._iterate()

            # increment iteration number
            self.nIt += 1

        # as a last, if there were any constraints to be taken care of,
        # we project the unconstrained variables back on the original constrained ones
        if self.constraints:
            self.constrain()
        
        # return the found solution functions
        return self.eqs.trajectories.x, self.eqs.trajectories.u

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

        # Initialise the spline function objects
        self.eqs.trajectories.init_splines()
        
        # Get an initial value (guess)
        self.eqs.get_guess()
        
        # Build the collocation equations system
        #G, DG = self.eqs.build(sys=self, trajectories=self.trajectories)
        C = self.eqs.build()
        G, DG = C.G, C.DG
        
        # Solve the collocation equation system
        sol = self.eqs.solve(G, DG)
        
        # Set the found solution
        self.eqs.trajectories.set_coeffs(sol)

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
        T = self.dyn_sys.b - self.dyn_sys.a
        
        # get list of start values
        start = []

        if self.constraints is not None:
            sys = self._dyn_sys_orig
        else:
            sys = self.dyn_sys
            
        x_vars = sys.states
        start_dict = dict([(k, v[0]) for k, v in sys.boundary_values.items() if k in x_vars])
        ff = sys.f_num
        
        for x in x_vars:
            start.append(start_dict[x])
        
        # create simulation object
        S = Simulator(ff, T, start, self.eqs.trajectories.u)
        
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
            bv = self._dyn_sys_orig.boundary_values
            x_sym = self._dyn_sys_orig.states 
        else:
            bv = self.dyn_sys.boundary_values
            x_sym = self.dyn_sys.states
            
        xb = dict([(k, v[1]) for k, v in bv.items() if k in x_sym])
        
        # what is the error
        logging.debug(40*"-")
        logging.debug("Ending up with:   Should Be:  Difference:")

        err = np.empty(xt.shape[1])
        for i, xx in enumerate(x_sym):
            err[i] = abs(xb[xx] - xt[-1][i])
            logging.debug(str(xx)+" : %f     %f    %f"%(xt[-1][i], xb[xx], err[i]))
        
        logging.debug(40*"-")
        
        #if self._ierr:
        ierr = self._parameters['ierr']
        eps = self._parameters['eps']
        if ierr:
            # calculate maximum consistency error on the whole interval
            maxH = auxiliary.consistency_error((a,b), self.eqs.trajectories.x, self.eqs.trajectories.u, self.eqs.trajectories.dx, self.dyn_sys.f_num)
            
            reached_accuracy = (maxH < ierr) and (max(err) < eps)
            logging.debug('maxH = %f'%maxH)
        else:
            # just check if tolerance for the boundary values is satisfied
            reached_accuracy = max(err) < eps
        
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

        if self.constraints:
            sys = self._dyn_sys_orig
        else:
            sys = self.dyn_sys
            
        # calculate the error functions H_i(t)
        max_con_err, error = auxiliary.consistency_error((sys.a, sys.b), 
                                                          self.eqs.trajectories.x,
                                                          self.eqs.trajectories.u, 
                                                          self.eqs.trajectories.dx, 
                                                          sys.f_num, len(self.sim_data[0]), True)
        
        H = dict()
        for i in self.eqs.trajectories._eqind:
            H[i] = error[:,i]

        visualisation.plot_simulation(self.sim_data, H)

    def save(self, fname=None):
        '''
        Save data using the python module :py:mod:`pickle`.
        
        The created pickle dumpfile contains the latest simulation data, i.e.
        a list of three arrays. The 1st entry is an array with the time steps
        of the simulation, the 2nd contains the corresponding values of the
        state variables and the 3rd those of the input variables.
        '''

        save = dict.fromkeys(['sys', 'eqs', 'traj'])

        # system state
        save['sys'] = dict()
        save['sys']['state'] = dict.fromkeys(['nIt', 'reached_accuracy'])
        save['sys']['state']['nIt'] = self.nIt
        save['sys']['state']['reached_accuracy'] = self.reached_accuracy
        
        # simulation results
        save['sys']['sim_data'] = self.sim_data

        # parameters
        save['sys']['parameters'] = self._parameters

        save['eqs'] = self.eqs.save()
        save['traj'] = self.eqs.trajectories.save()
        
        if fname is not None:
            if not (fname.endswith('.pcl') or fname.endswith('.pcl')):
                fname += '.pcl'
        
            with open(fname, 'w') as dumpfile:
                pickle.dump(save, dumpfile)

        return save

class DynamicalSystem(object):
    '''
    Provides access to information about the dynamical system that is the
    object of the control process.

    Parameters
    ----------

    f_sym : callable
        The (symbolic) vector field of the dynamical system

    a, b : floats
        The initial end final time of the control process

    xa, xb : iterables
        The initial and final conditions for the state variables

    ua, ub : iterables
        The initial and final conditions for the input variables
    '''

    def __init__(self, f_sym, a=0., b=1., xa=[], xb=[], ua=[], ub=[]):
        self.f_sym = f_sym
        self.a = a
        self.b = b

        # analyse the given system
        self.n_states, self.n_inputs = self._determine_system_dimensions(n=len(xa))

        # set names of the state and input variables
        # (will be used as keys in various dictionaries)
        self.states = tuple(['x{}'.format(i+1) for i in xrange(self.n_states)])
        self.inputs = tuple(['u{}'.format(j+1) for j in xrange(self.n_inputs)])
        
        # init dictionary for boundary values
        self.boundary_values = self._get_boundary_dict_from_lists(xa, xb, ua, ub)

        # create a numeric counterpart for the vector field
        # for faster evaluation
        self.f_num = auxiliary.sym2num_vectorfield(f_sym=self.f_sym, x_sym=self.states, u_sym=self.inputs,
                                                   vectorized=False, cse=False)

    def _determine_system_dimensions(self, n):
        '''
        Determines the number of state and input variables.

        Parameters
        ----------

        n : int
            Length of the list of initial state values
        '''

        # first, determine system dimensions
        logging.debug("Determine system/input dimensions")
        
        # the number of system variables can be determined via the length
        # of the boundary value lists
        n_states = n
        
        # now we want to determine the input dimension
        # therefore we iteratively increase the inputs dimension and try to call
        # the vectorfield
        found_n_inputs = False
        x = np.ones(n_states)

        j = 0
        while not found_n_inputs:
            u = np.ones(j)

            try:
                self.f_sym(x, u)
                # if no ValueError is raised j is the dimension of the inputs
                n_inputs = j
                found_n_inputs = True
            except (TypeError, ValueError):
                # unpacking error inside f_sym
                # (that means the dimensions don't match)
                j += 1
        
        logging.debug("--> state: {}".format(n_states))
        logging.debug("--> input : {}".format(n_inputs))

        return n_states, n_inputs

    def _get_boundary_dict_from_lists(self, xa, xb, ua, ub):
        '''
        Creates a dictionary of boundary values for the state and input variables
        for easier access.
        '''

        # consistency check
        assert len(xa) == len(xb) == self.n_states
        #assert len(ua) == len(ub) == self.n_inputs
        if not ua and not ub:
            ua = [None] * self.n_inputs
            ub = [None] * self.n_inputs

        # init dictionary
        boundary_values = dict()

        # add state boundary values
        for i, x in enumerate(self.states):
            boundary_values[x] = (xa[i], xb[i])

        # add input boundary values
        for j, u in enumerate(self.inputs):
            boundary_values[u] = (ua[j], ub[j])

        return boundary_values
