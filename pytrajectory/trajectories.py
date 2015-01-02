# IMPORTS
import numpy as np
import logging

from splines import ConstantSpline, LinearSpline, QuadraticSpline, CubicSpline
import auxiliary

from IPython import embed as IPS


class Trajectory(object):
    '''
    This class handles the creation and managing of the
    spline functions that are intended to approximate
    the desired trajectories.
    
    Parameters
    ----------
    
    ctrl_sys : system.ControlSystem
        Instance of a control system providing information like
        vector field function, integrator chains, boundary values
        and so on.
    '''
    
    def __init__(self, sys):
        # Save the control system instance
        # TODO: get rid of this need
        self.sys = sys
        
        # Save some information about the control system
        self._a = sys.a
        self._b = sys.b
        self._x_sym = sys.x_sym
        self._u_sym = sys.u_sym
        self._chains = sys.chains
        self._spline_orders = sys.mparam['spline_order']
        
        # Initialise dictionaries as containers for all
        # spline functions that will be created
        self._splines = dict()
        self._x_fnc = dict()
        self._u_fnc = dict()
        self._dx_fnc = dict()
        
        # This will be the free parameters of the control problem
        # (list of all independent spline coefficients)
        self.indep_coeffs = []
        
        self._old_splines = None
    
    
    def x(self, t):
        '''
        Returns the current system state.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the system at.
        '''
        
        if not self._a <= t <= self._b:
            logging.warning("Time point 't' has to be in (a,b)")
            arr = None
        else:
            arr = np.array([self._x_fnc[xx](t) for xx in self._x_sym])
        
        return arr


    def u(self, t):
        '''
        Returns the state of the input variables.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the input variables at.
        '''
        
        if not self._a <= t <= self._b:
            logging.warning("Time point 't' has to be in (a,b)")
            arr = np.array([self._u_fnc[uu](self._b) for uu in self._u_sym])
        else:
            arr = np.array([self._u_fnc[uu](t) for uu in self._u_sym])
        
        return arr


    def dx(self, t):
        '''
        Returns the state of the 1st derivatives of the system variables.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the 1st derivatives at.
        '''
        
        if not self._a <= t <= self._b:
            logging.warning("Time point 't' has to be in (a,b)")
            arr = None
        else:
            arr = np.array([self._dx_fnc[xx](t) for xx in self._x_sym])
        
        return arr
    
    
    
    def init_splines(self, sx, su, boundary_values, use_chains, spline_orders):
        '''
        This method is used to create the necessary spline function objects.
        
        Parameters
        ----------
        
        sx : int
            Number of polynomial parts for the state spline functions
        
        su : int
            Number of polynomial parts for the input spline functions
        
        boundary_values : dict
            Dictionary of boundary values for the state and input splines functions.
        
        use_chains : bool
            Whether or not to make use of system structure (integrator chains).
        
        '''
        logging.debug("Initialise Splines")
        
        bv = boundary_values
        
        # dictionaries for splines and callable solution function for x,u and dx
        splines = dict()
        x_fnc = dict()
        u_fnc = dict()
        dx_fnc = dict()
        
        #spline_classes = [ConstantSpline, LinearSpline, QuadraticSpline, CubicSpline]
        spline_classes = [LinearSpline, LinearSpline, QuadraticSpline, CubicSpline]
        
        if use_chains:
            # first handle variables that are part of an integrator chain
            for chain in self._chains:
                upper = chain.upper
                lower = chain.lower
        
                # here we just create a spline object for the upper ends of every chain
                # w.r.t. its lower end (whether it is an input variable or not)
                if chain.lower.name.startswith('x'):
                    splines[upper] = CubicSpline(self._a, self._b, n=sx, bc={0:bv[upper]}, tag=upper.name)
                    splines[upper].type = 'x'
                elif chain.lower.name.startswith('u'):
                    splines[upper] = CubicSpline(self._a, self._b, n=su, bc={0:bv[lower]}, tag=upper.name)
                    splines[upper].type = 'u'
        
                # search for boundary values to satisfy
                for i, elem in enumerate(chain.elements()):
                    if elem in self._x_sym:
                        splines[upper].boundary_values(i, boundary_values[elem])
                        if splines[upper].type == 'u':
                            splines[upper].boundary_values(i+1, boundary_values[lower])
        
                # solve smoothness and boundary conditions
                splines[upper].make_steady()
        
                # calculate derivatives
                for i, elem in enumerate(chain.elements()):
                    if elem in self._x_sym:
                        x_fnc[elem] = splines[upper].derive(i)
                    elif elem in self._u_sym:
                        u_fnc[elem] = splines[upper].derive(i)

        # now handle the variables which are not part of any chain
        for i, xx in enumerate(self._x_sym):
            if (not x_fnc.has_key(xx)):
                #splines[xx] = CubicSpline(self._a, self._b, n=sx, bc={0:bv[xx]}, tag=xx.name, steady=True)
                SplineClass = spline_classes[spline_orders[i]]
                splines[xx] = SplineClass(self._a, self._b, n=sx, bc={0:bv[xx]}, tag=xx.name, steady=True)
                splines[xx].type = 'x'
                x_fnc[xx] = splines[xx]
        
        offset = len(self._x_sym)
        for j, uu in enumerate(self._u_sym):
            if (not u_fnc.has_key(uu)):
                #splines[uu] = CubicSpline(self._a, self._b, n=su, bc={0:bv[uu]}, tag=uu.name, steady=True)
                SplineClass = spline_classes[spline_orders[offset+j]]
                splines[uu] = SplineClass(self._a, self._b, n=su, bc={0:bv[uu]}, tag=uu.name, steady=True)
                splines[uu].type = 'u'
                u_fnc[uu] = splines[uu]
    
        # calculate derivatives of every state variable spline
        for xx in self._x_sym:
            dx_fnc[xx] = x_fnc[xx].derive()

        indep_coeffs = dict()
        for ss in splines:
            indep_coeffs[ss] = splines[ss]._indep_coeffs
        
        self.indep_coeffs = indep_coeffs
        self._splines = splines
        self._x_fnc = x_fnc
        self._u_fnc = u_fnc
        self._dx_fnc = dx_fnc
        
    
    def set_coeffs(self, sol, use_chains):
        '''
        Set found numerical values for the independent parameters of each spline.

        This method is used to get the actual splines by using the numerical
        solutions to set up the coefficients of the polynomial spline parts of
        every created spline.
        
        Parameters
        ----------
        
        sol : numpy.ndarray
            The solution vector for the free parameters, i.e. the independent coefficients.
        
        '''
        
        logging.debug("Set spline coefficients")
        
        sol_bak = sol.copy()
        subs = dict()

        for k, v in sorted(self.indep_coeffs.items(), key=lambda (k, v): k.name):
            i = len(v)
            subs[k] = sol[:i]
            sol = sol[i:]
        
        if use_chains:
            for var in self._x_sym + self._u_sym:
                for ic in self._chains:
                    if var in ic:
                        subs[var] = subs[ic.upper]
        
        # set numerical coefficients for each spline and derivative
        for k, v in self._x_fnc.items() + self._u_fnc.items() + self._dx_fnc.items():
            v.set_coefficients(free_coeffs=subs[k])
        
        # yet another dictionary for solution and coeffs
        coeffs_sol = dict()

        # used for indexing
        i = 0
        j = 0

        for k, v in sorted(self.indep_coeffs.items(), key=lambda (k, v): k.name):
            j += len(v)
            coeffs_sol[k] = sol_bak[i:j]
            i = j

        self.coeffs_sol = coeffs_sol
    
    
    def check_accuracy(self, sim_data, ff, x_sym, boundary_values):
        '''
        Checks whether the desired accuracy for the boundary values was reached.

        It calculates the difference between the solution of the simulation
        and the given boundary values at the right border and compares its
        maximum against the tolerance.
        
        If set by the user it also calculates some kind of consistency error
        that shows how "well" the spline functions comply with the system
        dynamic given by the vector field.
        
        Parameters
        ----------
        
        sim_data : tuple
            Contains collocation points, and simulation results of system and input variables.
        
        ff : callable
            The vector field of the control system.
        
        x_sym : iterable
            Sympy.symbols of the state variables.
        
        boundary_values : dict
            Dictionary of boundary values for the state and input splines functions.
        
        Returns
        -------
        
        bool
            Whether the desired tolerance is satisfied or not.
        
        '''
        
        # this is the solution of the simulation
        a = sim_data[0][0]
        b = sim_data[0][-1]
        xt = sim_data[1]
        
        # get boundary values at right border of the interval
        xb = dict([(k, v[1]) for k, v in boundary_values.items() if k in x_sym])
        
        # what is the error
        logging.debug(40*"-")
        logging.debug("Ending up with:   Should Be:  Difference:")

        err = np.empty(xt.shape[1])
        for i, xx in enumerate(x_sym):
            err[i] = abs(xb[xx] - xt[-1][i])
            logging.debug(str(xx)+" : %f     %f    %f"%(xt[-1][i], xb[xx], err[i]))
        
        logging.debug(40*"-")
        
        if self.sys.mparam['ierr']:
            # calculate maximum consistency error on the whole interval
            maxH = auxiliary.consistency_error((a,b), self.x, self.u, self.dx, ff)
            
            reached_accuracy = (maxH < self.sys.mparam['ierr']) and (max(err) < self.sys.mparam['eps'])
            logging.debug('maxH = %f'%maxH)
        else:
            # just check if tolerance for the boundary values is satisfied
            reached_accuracy = max(err) < self.sys.mparam['eps']
        
        if reached_accuracy:
            logging.info("  --> reached desired accuracy: "+str(reached_accuracy))
        else:
            logging.debug("  --> reached desired accuracy: "+str(reached_accuracy))
        
        return reached_accuracy
    
        