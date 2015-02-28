# IMPORTS
import numpy as np
import sympy as sp
from scipy import sparse

from log import logging, Timer
from solver import Solver

from auxiliary import sym2num_vectorfield

from IPython import embed as IPS


class CollocationSystem(object):
    '''
    This class represents the collocation system that is used
    to determine a solution for the free parameters of the
    control system, i.e. the independent coefficients of the
    trajectory splines.
    
    Parameters
    ----------
    
    sys : system.ControlSystem
        Instance of a control system.
    
    '''
    def __init__(self, sys, tol=1e-5, sol_steps=100, method='leven', coll_type='equidistant', use_sparse=True):
        # TODO: get rid of the following
        self.sys = sys
        
        # Save some information
        self._tol = tol
        self._sol_steps = sol_steps
        self._method = method
        self._coll_type = coll_type
        self._use_sparse = use_sparse
        
        # we don't have a soution, yet
        self.sol = None
        
        # create vectorized versions of the control system's vector field
        # and its jacobian for the faster evaluation of the collocation equation system `G`
        # and its jacobian `DG` (--> see self.build())
        #f = sys.ff_sym(sp.symbols(sys.x_sym), sp.symbols(sys.u_sym))
        f = sys.ff_sym(sys.x_sym, sys.u_sym)
        Df = sp.Matrix(f).jacobian(sys.x_sym+sys.u_sym)
        
        self._ff_vectorized = sym2num_vectorfield(f, sys.x_sym, sys.u_sym, vectorized=True)
        self._Df_vectorized = sym2num_vectorfield(Df, sys.x_sym, sys.u_sym, vectorized=True)
    
    def build(self, sys, trajectories):
        '''
        This method is used to set up the equations for the collocation equation system
        and defines functions for the numerical evaluation of the system and its jacobian.
        
        Parameters
        ----------
        
        sys : pytrajectory.system.ControlSystem
            The control system for which to build the collocation equation system.
        
        trajectories : pytrajectory.trajectories.Trajectory
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
        x_fnc = trajectories._x_fnc
        dx_fnc = trajectories._dx_fnc
        u_fnc = trajectories._u_fnc

        # make symbols local
        x_sym = trajectories._x_sym
        u_sym = trajectories._u_sym

        a = sys.a
        b = sys.b
        delta = 2#sys.mparam['delta']

        # now we generate the collocation points
        cpts = collocation_nodes(a=a, b=b, npts=(trajectories._sx * delta + 1), coll_type=self._coll_type)
    
        # here we do something that will be explained after we've done it  ;-)
        indic = dict()
        i = 0
        j = 0
    
        # iterate over spline quantities
        for k, v in sorted(trajectories.indep_coeffs.items(), key=lambda (k, v): k.name):
            # increase j by the number of indep coeffs on which it depends
            j += len(v)
            indic[k] = (i, j)
            i = j
    
        # iterate over all quantities including inputs
        # and take care of integrator chain elements
        if trajectories._use_chains:
            for sq in x_sym+u_sym:
                for ic in trajectories._chains:
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
        free_param = np.hstack(sorted(trajectories.indep_coeffs.values(), key=lambda arr: arr[0].name))
        c_len = free_param.size
        
        with Timer('Building M-matrices'):
            lx = len(cpts)*len(x_sym)
            lu = len(cpts)*len(u_sym)

            Mx = [None]*lx
            Mx_abs = [None]*lx
            Mdx = [None]*lx
            Mdx_abs = [None]*lx
            Mu = [None]*lu
            Mu_abs = [None]*lu

            eqx = 0
            equ = 0
            for p in cpts:
                for xx in x_sym:
                    mx = np.zeros(c_len)
                    mdx = np.zeros(c_len)

                    i,j = indic[xx]

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
        
        # the following would be created with every call to self.DG but it is possible to
        # only do it once. So we do it here to speed things up.

        # here we compute the jacobian matrix of the derivatives of the system state functions
        # (as they depend on the free parameters in a linear fashion its just the above matrix Mdx)
        DdX = Mdx.reshape((len(cpts),-1,free_param.size))
        if trajectories._use_chains:
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
        
        if self._use_sparse:
            Mx = sparse.csr_matrix(Mx)
            Mx_abs = sparse.csr_matrix(Mx_abs)
            Mdx = sparse.csr_matrix(Mdx)
            Mdx_abs = sparse.csr_matrix(Mdx_abs)
            Mu = sparse.csr_matrix(Mu)
            Mu_abs = sparse.csr_matrix(Mu_abs)

            DdX = sparse.csr_matrix(DdX)
            DXU = sparse.csr_matrix(DXU)
        
        # localize vectorized functions for the control system's vector field and its jacobian
        ff_vec = self._ff_vectorized
        Df_vec = self._Df_vectorized
        
        # in the later evaluation of the equation system `G` and its jacobian `DG`
        # there will be created the matrices `F` and DF in which every `x` rows represent the 
        # evaluation of the control systems vectorfield and its jacobian in a specific collocation
        # point, where `x` is the number of state variables
        # 
        # if we make use of the system structure, i.e. the integrator chains, not every
        # equation of the vector field has to be solved and because of that, not every row 
        # of the matrices `F` and `DF` is neccessary
        # 
        # therefore we now create an array with the indices of all rows we need from these matrices
        if trajectories._use_chains:
            eqind = sys.eqind
        else:
            eqind = range(len(sys.x_sym))
        
        # `eqind` now contains the indices of the equations/rows of the vector field
        # that have to be solved
        
        cp_len = len(cpts)
        
        # this (-> `take_indices`) will be the array with indices of the rows we need
        # 
        # to get these indices we iterate over all rows and take those whose indices
        # are contained in `eqind` (module the number of state variables -> `x_len`)
        take_indices = np.array([idx for idx in xrange(cp_len*x_len) if idx%x_len in eqind])
        
        # define the callable functions for the eqs
        def G(c):
            X = Mx.dot(c) + Mx_abs
            U = Mu.dot(c) + Mu_abs
            
            X = np.array(X).reshape((x_len, -1), order='F')
            U = np.array(U).reshape((u_len, -1), order='F')
        
            # evaluate system equations and select those related
            # to lower ends of integrator chains (via eqind)
            # other equations need not to be solved
            F = ff_vec(X, U).take(eqind, axis=0)
        
            dX = Mdx.dot(c) + Mdx_abs
            dX = np.array(dX).reshape((x_len, -1), order='F').take(eqind, axis=0)
    
            G = F - dX
            
            return G.ravel(order='F')
    
        # and its jacobian
        def DG(c):
            '''
            Returns the jacobian matrix of the collocation system w.r.t. the
            independent parameters evaluated at :attr:`c`.
            '''
        
            # first we calculate the x and u values in all collocation points
            # with the current numerical values of the free parameters
            X = Mx.dot(c) + Mx_abs
            X = np.array(X).reshape((x_len, -1), order='F')
            U = Mu.dot(c) + Mu_abs
            U = np.array(U).reshape((u_len, -1), order='F')
            
            # get the jacobian blocks and turn them into the right shape
            DF_blocks = Df_vec(X,U).swapaxes(0,2).swapaxes(1,2)

            # build a block diagonal matrix from the blocks
            DF_csr = sparse.block_diag(DF_blocks, format='csr').dot(DXU)
        
            # if we make use of the system structure
            # we have to select those rows which correspond to the equations
            # that have to be solved
            if trajectories._use_chains:
                DF_csr = sparse.csr_matrix(DF_csr.toarray().take(take_indices, axis=0))
                # TODO: is the performance gain that results from not having to solve
                #       some equations (use integrator chains) greater than
                #       the performance loss that results from transfering the
                #       sparse matrix to a full numpy array and back to a sparse matrix?
        
            DG = DF_csr - DdX
        
            return DG
        
        # return the callable functions
        return G, DG
    
    
    def get_guess(self, trajectories):
        '''
        This method is used to determine a starting value (guess) for the
        solver of the collocation equation system.

        If it is the first iteration step, then a vector with the same length as
        the vector of the free parameters with arbitrary values is returned.

        Else, for every variable a spline has been created for, the old spline
        of the iteration before and the new spline are evaluated at specific
        points and a equation system is solved which ensures that they are equal
        in these points.

        The solution of this system is the new start value for the solver.
        
        Parameters
        ----------
        
        trajectories : pytrajectory.trajectories.Trajectory
        
        '''
        if not trajectories._old_splines:
            free_coeffs_all = np.hstack(trajectories.indep_coeffs.values())
            guess = 0.1 * np.ones(free_coeffs_all.size)
        else:
            guess = np.empty(0)
            
            # get new guess for every independent variable
            for k, v in sorted(trajectories.indep_coeffs.items(), key = lambda (k, v): k.name):
                if (trajectories._splines[k].type == 'x'):
                    logging.debug("Get new guess for spline {}".format(k))
                    
                    s_new = trajectories._splines[k]
                    s_old = trajectories._old_splines[k]
                    
                    # how many independent coefficients does the spline have
                    coeffs_size = s_new._indep_coeffs.size
                    
                    # generate points to evaluate the old spline at
                    # (new and old spline should be equal in these)
                    #guess_points = np.linspace(s_new.a, s_new.b, coeffs_size, endpoint=False)
                    guess_points = np.linspace(s_new.a, s_new.b, coeffs_size, endpoint=True)
                    
                    # evaluate the splines
                    s_old_t = np.array([s_old.f(t) for t in guess_points])
                    
                    dep_vecs = [s_new.get_dependence_vectors(t) for t in guess_points]
                    s_new_t = np.array([vec[0] for vec in dep_vecs])
                    s_new_t_abs = np.array([vec[1] for vec in dep_vecs])
                    
                    #old_style_guess = np.linalg.solve(s_new_t, s_old_t - s_new_t_abs)
                    old_style_guess = np.linalg.lstsq(s_new_t, s_old_t - s_new_t_abs)[0]
                    
                    guess = np.hstack((guess, old_style_guess))
                        
                else:
                    # if it is a input variable, just take the old solution
                    guess = np.hstack((guess, trajectories._old_splines[k]._indep_coeffs))
        
        if 0:
            
            try:
                import matplotlib.pyplot as plt
            
                tt = np.linspace(trajectories._a, trajectories._b, 1000)
                xt_old = np.zeros((1000,len(trajectories._x_sym)))
            
                for i, x in enumerate(trajectories._x_sym):
                    fx = trajectories._old_splines[x]
                    xt_old[:,i] = [fx(t) for t in tt]
            
                sol_bak = 1.0*self.sol
                splines_bak = trajectories._splines.copy()
            
                trajectories.set_coeffs(guess)
                xt_new = np.zeros((1000,len(trajectories._x_sym)))
                
                for i, x in enumerate(trajectories._x_sym):
                    fx = trajectories._splines[x]
                    xt_new[:,i] = [fx(t) for t in tt]
                
                print np.abs(xt_old-xt_new).max()
                IPS()
                
                self.sol = sol_bak
                trajectories._splines = splines_bak
                for s in trajectories._splines.values():
                    s._prov_flag = True
                
            except Exception as err:
                print "guess plot error"
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
        solver = Solver(F=G, DF=DG, x0=self.guess, tol=self._tol,
                        maxIt=self._sol_steps, method=self._method)
        
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
    
    
        