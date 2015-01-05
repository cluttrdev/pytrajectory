# IMPORTS
import numpy as np
import sympy as sp
from scipy import sparse
import logging

from solver import Solver

from splines import interpolate

from IPython import embed as IPS


class CollocationSystem(object):
    '''
    This class represents the collocation system that is used
    to determine a solution for the free parameters of the
    control system, i.e. the independent coefficients of the
    trajectory splines.
    
    Parameters
    ----------
    
    CtrlSys : system.ControlSystem
        Instance of a control system.
    
    '''
    def __init__(self, ctrl_sys):
        # TODO: get rid of the following
        self.sys = ctrl_sys
        
        # Save some information
        self._coll_type = ctrl_sys.mparam['coll_type']
        self._use_sparse = ctrl_sys.mparam['use_sparse']
        self.sol = 0
    
    
    def build(self, sys, traj):
        '''
        This method is used to set up the equations for the collocation equation system
        and defines functions for the numerical evaluation of the system and its jacobian.
        
        Parameters
        ----------
        
        sys : system.ControlSystem
            The control system for which to build the collocation equation system.
        
        traj : trajectories.Trajectory
            The control system's trajectory object.
        
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
        x_fnc = traj._x_fnc
        dx_fnc = traj._dx_fnc
        u_fnc = traj._u_fnc

        # make symbols local
        x_sym = traj._x_sym
        u_sym = traj._u_sym

        a = sys.a
        b = sys.b
        delta = sys.mparam['delta']

        # now we generate the collocation points
        cpts = collocation_nodes(a=a, b=b, npts=(sys.mparam['sx']*delta+1), coll_type=sys.mparam['coll_type'])
        
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
        for k, v in sorted(sys.trajectories.indep_coeffs.items(), key=lambda (k, v): k.name):
            # increase j by the number of indep coeffs on which it depends
            j += len(v)
            indic[k] = (i, j)
            i = j
        
        # iterate over all quantities including inputs
        # and take care of integrator chain elements
        if sys.mparam['use_chains']:
            for sq in x_sym+u_sym:
                for ic in sys.chains:
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

                i,j = indic[xx]

                mx[i:j], Mx_abs[eqx] = x_fnc[xx].get_dependence_vectors(p)
                mdx[i:j], Mdx_abs[eqx] = dx_fnc[xx].get_dependence_vectors(p)

                Mx[eqx] = mx
                Mdx[eqx] = mdx
                eqx += 1

            for uu in u_sym:
                mu = np.zeros(c_len)

                i,j = indic[uu]
                
                mu[i:j], Mu_abs[equ] = u_fnc[uu].get_dependence_vectors(p)
                
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
        f = sys.ff_sym(x_sym,u_sym)
        Df_mat = sp.Matrix(f).jacobian(x_sym+u_sym)
        Df = sp.lambdify(x_sym+u_sym, Df_mat, modules='numpy')

        # the following would be created with every call to self.DG but it is possible to
        # only do it once. So we do it here to speed things up.

        # here we compute the jacobian matrix of the derivatives of the system state functions
        # (as they depend on the free parameters in a linear fashion its just the above matrix Mdx)
        DdX = Mdx.reshape((len(cpts),-1,len(self.c_list)))
        if sys.mparam['use_chains']:
            DdX = DdX[:,sys.eqind,:]
        DdX = np.vstack(DdX)

        # here we compute the jacobian matrix of the system/input functions as they also depend on
        # the free parameters
        DXU = []
        x_len = len(sys.x_sym)
        u_len = len(sys.u_sym)
        xu_len = x_len + u_len

        for i in xrange(len(cpts)):
            DXU.append(np.vstack(( Mx[x_len*i:x_len*(i+1)], Mu[u_len*i:u_len*(i+1)] )))
        DXU_old = DXU
        DXU = np.vstack(DXU)
        
        if sys.mparam['use_sparse']:
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
        ff = sys.ff
        
        if sys.mparam['use_chains']:
            eqind = sys.eqind
        else:
            eqind = range(len(sys.x_sym))
        
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
            
            #if sys.mparam['sx'] == 40:
            #    print "G: sx=40 nan"
            #    IPS()
            
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
            if sys.mparam['use_chains']:
                DF_csr = [row for idx,row in enumerate(DF_csr.toarray()[:]) if idx%x_len in eqind]
                DF_csr = sparse.csr_matrix(DF_csr)
            
            DG = DF_csr - DdX
            
            return DG
        
        #if sys.mparam['sx'] == 40:
        #    print "G: sx=40 nan"
        #    IPS()
        
        # return the callable functions
        return G, DG
    
    
    def get_guess(self, free_coeffs, old_splines, new_splines):
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
        
        Parameters
        ----------
        
        free_coeffs : dict
            The free parameters, i.e. the independent coefficients of the splines.
        
        old_splines : dict
            The splines from the last iteration (None if there are none).
        
        new_splines : dict
            The newly initialised splines of the current iteration step.
        
        '''
        if (old_splines == None):
            self.c_list = np.empty(0)

            for k, v in sorted(free_coeffs.items(), key = lambda (k, v): k.name):
                self.c_list = np.hstack((self.c_list, v))
            guess = 0.1*np.ones(len(self.c_list))
        else:
            guess = np.empty(0)
            self.c_list = np.empty(0)

            # get new guess for every independent variable
            #for k, v in sorted(self.sys.trajectories.coeffs_sol.items(), key = lambda (k, v): k.name):
            for k, v in sorted(free_coeffs.items(), key = lambda (k, v): k.name):
                self.c_list = np.hstack((self.c_list, free_coeffs[k]))

                if (new_splines[k].type == 'x'):
                    logging.debug("Get new guess for spline %s"%k.name)
                    
                    s_new = new_splines[k]
                    s_old = old_splines[k]
                    s_interp = interpolate(fnc=s_old, points=s_new.nodes)
                    
                    # get indices of new independent coefficients
                    idx = [(int(i),int(j)) for i,j in [coeff.name.split('_')[-2:] for coeff in s_new._indep_coeffs]]
                    
                    # get values for them from the interpolant
                    interp_guess = np.array([s_interp._coeffs[ij] for ij in idx])
                    
                    guess = np.hstack((guess,interp_guess))
                else:
                    # if it is a input variable, just take the old solution
                    #guess = np.hstack((guess, self.sys.trajectories.coeffs_sol[k]))
                    guess = np.hstack((guess, old_splines[k]._indep_coeffs))
        
        if 1:
            
            try:
                import matplotlib.pyplot as plt
            
                tt = np.linspace(self.sys.a, self.sys.b, 1000)
                xt_old = np.zeros((1000,len(self.sys.x_sym)))
            
                for i, x in enumerate(self.sys.x_sym):
                    fx = old_splines[x]
                    xt_old[:,i] = [fx(t) for t in tt]
            
                sol_bak = 1.0*self.sol
                splines_bak = new_splines.copy()
            
                self.sys.trajectories.set_coeffs(guess, False)
                xt_new = np.zeros((1000,len(self.sys.x_sym)))
                
                for i, x in enumerate(self.sys.x_sym):
                    fx = new_splines[x]
                    xt_new[:,i] = [fx(t) for t in tt]
                
                
                print np.abs(xt_old-xt_new).max()
                IPS()
                
                self.sol = sol_bak
                new_splines = splines_bak
                for s in new_splines.values():
                    s._prov_flag = True
                
            except Exception as err:
                IPS()
                pass
        
        # the new guess
        self.guess = guess
    
    
    def solve(self, G, DG):
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
        solver = Solver(F=G, DF=DG, x0=self.guess, tol=self.sys.mparam['tol'],
                        maxIt=self.sys.mparam['sol_steps'], method=self.sys.mparam['method'])
        
        # solve the equation system
        self.sol = solver.solve()
        
        #print "SOLVE"
        #IPS()
        
        #from scipy import optimize as op
        #scipy_sol = op.root(fun=self.G, x0=self.guess, method='lm', jac=False)
        #IPS()
        
        return self.sol


def collocation_nodes(a, b, npts, coll_type):
    '''
    Create collocation points/nodes for the equation system.
    
    Parameters
    ----------
    
    a : float
        The left border of the considered interval.
    
    b : float
        The right border of the considered interval.
    
    npts : int
        The number of nodes.
    
    coll_type : str
        Specifies how to generate the nodes.
    
    Returns
    -------
    
    numpy.ndarray
        The collocation nodes.
    
    '''
    if coll_type == 'equidistant':
        # get equidistant collocation points
        cpts = np.linspace(a, b, npts, endpoint=True)
    elif coll_type == 'chebychev':
        # determine rank of chebychev polynomial
        # of which to calculate zero points
        nc = int(npts) - 2

        # calculate zero points of chebychev polynomial --> in [-1,1]
        cheb_cpts = [np.cos( (2.0*i+1)/(2*(nc+1)) * np.pi) for i in xrange(nc)]
        cheb_cpts.sort()

        # transfer chebychev nodes from [-1,1] to our interval [a,b]
        chpts = [a + (b-a)/2.0 * (chp + 1) for chp in cheb_cpts]

        # add left and right borders
        cpts = np.hstack((a, chpts, b))
    else:
        logging.warning('Unknown type of collocation points.')
        logging.warning('--> will use equidistant points!')
        cpts = np.linspace(a, b, npts, endpoint=True)
    
    return cpts
    
    
        