# IMPORTS
import numpy as np
import sympy as sp
from scipy import sparse

from log import logging, Timer
from trajectories import Trajectory
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
        Instance of a control system
    
    tol : float
        Tolerance for the solver
    
    sol_steps : int
        Maximum number of steps of the nonlinear solver
    
    method : str
        The method to solve the nonlinear equation system
    
    coll_type : str
        The type of the collocation points
    
    '''
    def __init__(self, sys, **kwargs):
        # TODO: get rid of the following
        self.sys = sys
        
        # set parameters
        #self._tol = kwargs.get('tol', 1e-5)
        #self._sol_steps = kwargs.get('sol_steps', 100)
        #self._method = kwargs.get('method', 'leven')
        #self._coll_type = kwargs.get('coll_type', 'equidistant')
        self._parameters = dict()
        self._parameters['tol'] = kwargs.get('tol', 1e-5)
        self._parameters['sol_steps'] = kwargs.get('sol_steps', 100)
        self._parameters['method'] = kwargs.get('method', 'leven')
        self._parameters['coll_type'] = kwargs.get('coll_type', 'equidistant')
        
        # we don't have a soution, yet
        self.sol = None
        
        # create vectorized versions of the control system's vector field
        # and its jacobian for the faster evaluation of the collocation equation system `G`
        # and its jacobian `DG` (--> see self.build())
        f = sys.ff_sym(sp.symbols(sys.x_sym), sp.symbols(sys.u_sym))
        
        # TODO: check order of variables of differentiation ([x,u] vs. [u,x])
        #       because in dot products in later evaluation of `DG` with vector `c`
        #       values for u come first in `c`
        Df = sp.Matrix(f).jacobian(sys.x_sym+sys.u_sym)
        #Df = sp.Matrix(f).jacobian(sys.u_sym+sys.x_sym)
        
        self._ff_vectorized = sym2num_vectorfield(f, sys.x_sym, sys.u_sym, vectorized=True, cse=True)
        self._Df_vectorized = sym2num_vectorfield(Df, sys.x_sym, sys.u_sym, vectorized=True, cse=True)
        self._f = f
        self._Df = Df

        self.trajectories = Trajectory(sys, **kwargs)

    def build(self):
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

        class Container(object):
            def __init__(self, **kwargs):
                for key, value in kwargs.iteritems():
                    self.__setattr__(str(key), value)

        logging.debug("Building Equation System")
        
        # make symbols local
        x_sym = self.trajectories._x_sym
        u_sym = self.trajectories._u_sym

        # determine for each spline the index range of its free coeffs in the concatenated
        # vector of all free coeffs
        indic = self._get_index_dict()

        # compute dependence matrices
        Mx, Mx_abs, Mdx, Mdx_abs, Mu, Mu_abs = self._build_dependence_matrices(indic)

        # in the later evaluation of the equation system `G` and its jacobian `DG`
        # there will be created the matrices `F` and DF in which every nx rows represent the 
        # evaluation of the control systems vectorfield and its jacobian in a specific collocation
        # point, where nx is the number of state variables
        # 
        # if we make use of the system structure, i.e. the integrator chains, not every
        # equation of the vector field has to be solved and because of that, not every row 
        # of the matrices `F` and `DF` is neccessary
        # 
        # therefore we now create an array with the indices of all rows we need from these matrices
        if self.trajectories._parameters['use_chains']:
            eqind = self.sys.eqind
        else:
            eqind = range(len(x_sym))

        # `eqind` now contains the indices of the equations/rows of the vector field
        # that have to be solved
        delta = 2
        n_cpts = self.trajectories.n_parts_x * delta + 1
        
        # this (-> `take_indices`) will be the array with indices of the rows we need
        # 
        # to get these indices we iterate over all rows and take those whose indices
        # are contained in `eqind` (modulo the number of state variables -> `x_len`)
        take_indices = np.tile(eqind, (n_cpts,)) + np.arange(n_cpts).repeat(len(eqind)) * len(x_sym)
        
        # here we determine the jacobian matrix of the derivatives of the system state functions
        # (as they depend on the free parameters in a linear fashion its just the above matrix Mdx)
        DdX = Mdx[take_indices, :]
        
        # here we compute the jacobian matrix of the system/input splines as they also depend on
        # the free parameters
        DXU = []
        x_len = len(x_sym)
        u_len = len(u_sym)
        xu_len = x_len + u_len

        for i in xrange(n_cpts):
            DXU.append(np.vstack(( Mx[x_len*i:x_len*(i+1)].toarray(), Mu[u_len*i:u_len*(i+1)].toarray() )))
            #DXU.append(np.vstack(( Mu[u_len*i:u_len*(i+1)].toarray(), Mx[x_len*i:x_len*(i+1)].toarray() )))
        DXU_old = DXU
        DXU = np.vstack(DXU)

        DXU = sparse.csr_matrix(DXU)

        # localize vectorized functions for the control system's vector field and its jacobian
        ff_vec = self._ff_vectorized
        Df_vec = self._Df_vectorized

        # transform matrix formats for faster dot products
        Mx = Mx.tocsr()
        Mx_abs = Mx_abs.tocsr()
        Mdx = Mdx.tocsr()
        Mdx_abs = Mdx_abs.tocsr()
        Mu = Mu.tocsr()
        Mu_abs = Mu_abs.tocsr()

        DdX = DdX.tocsr()
        
        # define the callable functions for the eqs
        def G(c):
            # TODO: check if both spline approaches result in same values here
            X = Mx.dot(c)[:,None] + Mx_abs
            U = Mu.dot(c)[:,None] + Mu_abs
            
            X = np.array(X).reshape((x_len, -1), order='F')
            U = np.array(U).reshape((u_len, -1), order='F')
        
            # evaluate system equations and select those related
            # to lower ends of integrator chains (via eqind)
            # other equations need not be solved
            #F = ff_vec(X, U).take(eqind, axis=0)
            F = ff_vec(X, U).ravel(order='F').take(take_indices, axis=0)[:,None]
        
            dX = Mdx.dot(c)[:,None] + Mdx_abs
            dX = dX.take(take_indices, axis=0)
            #dX = np.array(dX).reshape((x_len, -1), order='F').take(eqind, axis=0)
    
            G = F - dX

            return np.asarray(G).ravel(order='F')

        # and its jacobian
        def DG(c):
            # first we calculate the x and u values in all collocation points
            # with the current numerical values of the free parameters
            X = Mx.dot(c)[:,None] + Mx_abs
            X = np.array(X).reshape((x_len, -1), order='F')
            U = Mu.dot(c)[:,None] + Mu_abs
            U = np.array(U).reshape((u_len, -1), order='F')
            
            # get the jacobian blocks and turn them into the right shape
            DF_blocks = Df_vec(X,U).transpose([2,0,1])

            # build a block diagonal matrix from the blocks
            DF_csr = sparse.block_diag(DF_blocks, format='csr').dot(DXU)
        
            # if we make use of the system structure
            # we have to select those rows which correspond to the equations
            # that have to be solved
            if self.trajectories._parameters['use_chains']:
                DF_csr = sparse.csr_matrix(DF_csr.toarray().take(take_indices, axis=0))
                # TODO: is the performance gain that results from not having to solve
                #       some equations (use integrator chains) greater than
                #       the performance loss that results from transfering the
                #       sparse matrix to a full numpy array and back to a sparse matrix?
        
            DG = DF_csr - DdX
        
            return DG

        C = Container(G=G, DG=DG,
                      Mx=Mx, Mx_abs=Mx_abs,
                      Mu=Mu, Mu_abs=Mu_abs,
                      Mdx=Mdx, Mdx_abs=Mdx_abs,
                      guess=self.guess)
        
        # return the callable functions
        #return G, DG
        return C

    def _get_index_dict(self):
        # here we do something that will be explained after we've done it  ;-)
        indic = dict()
        i = 0
        j = 0
    
        # iterate over spline quantities
        for k, v in sorted(self.trajectories.indep_coeffs.items(), key=lambda (k, v): k):
            # increase j by the number of indep coeffs on which it depends
            j += len(v)
            indic[k] = (i, j)
            i = j
    
        # iterate over all quantities including inputs
        # and take care of integrator chain elements
        if self.trajectories._parameters['use_chains']:
            for sq in self.trajectories._x_sym + self.trajectories._u_sym:
                for ic in self.trajectories._chains:
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

        return indic

    def _build_dependence_matrices(self, indic):
        # first we compute the collocation points
        cpts = collocation_nodes(a=self.trajectories._a, b=self.trajectories._b,
                                 npts=self.trajectories.n_parts_x * 2 + 1,
                                 coll_type=self._parameters['coll_type'])

        x_fnc = self.trajectories.x_fnc
        dx_fnc = self.trajectories.dx_fnc
        u_fnc = self.trajectories.u_fnc

        x_sym = self.trajectories._x_sym
        u_sym = self.trajectories._u_sym
        
        # total number of independent coefficients
        free_param = np.hstack(sorted(self.trajectories.indep_coeffs.values(), key=lambda arr: arr[0].name))
        n_dof = free_param.size
        
        lx = len(cpts)*len(x_sym)
        lu = len(cpts)*len(u_sym)
        
        # initialize sparse dependence matrices
        Mx = sparse.lil_matrix((lx, n_dof))
        Mx_abs = sparse.lil_matrix((lx, 1))
        
        Mdx = sparse.lil_matrix((lx, n_dof))
        Mdx_abs = sparse.lil_matrix((lx, 1))
        
        Mu = sparse.lil_matrix((lu, n_dof))
        Mu_abs = sparse.lil_matrix((lu, 1))
        
        for ip, p in enumerate(cpts):
            for ix, xx in enumerate(x_sym):
                # get index range of `xx` in vector of all indep coeffs
                i,j = indic[xx]

                # determine derivation order according to integrator chains
                dorder_fx = _get_derivation_order(x_fnc[xx])
                dorder_dfx = _get_derivation_order(dx_fnc[xx])
                assert dorder_dfx == dorder_fx + 1

                # get dependence vector for the collocation point and spline variable
                mx, mx_abs = x_fnc[xx].im_self.get_dependence_vectors(p, d=dorder_fx)
                mdx, mdx_abs = dx_fnc[xx].im_self.get_dependence_vectors(p, d=dorder_dfx)

                k = ip * len(x_sym) + ix
                
                Mx[k, i:j] = mx
                Mx_abs[k] = mx_abs

                Mdx[k, i:j] = mdx
                Mdx_abs[k] = mdx_abs
                
            for iu, uu in enumerate(u_sym):
                # get index range of `xx` in vector of all indep coeffs
                i,j = indic[uu]

                dorder_fu = _get_derivation_order(u_fnc[uu])

                # get dependence vector for the collocation point and spline variable
                mu, mu_abs = u_fnc[uu].im_self.get_dependence_vectors(p, d=dorder_fu)

                k = ip * len(u_sym) + iu
                
                Mu[k, i:j] = mu
                Mu_abs[k] = mu_abs

        return Mx, Mx_abs, Mdx, Mdx_abs, Mu, Mu_abs

    def _build_dependence_jacobians(self, Mx, Mdx, Mu):

        
        
        
        # here we compute the jacobian matrix of the derivatives of the system state functions
        # (as they depend on the free parameters in a linear fashion its just the above matrix Mdx)
        DdX = Mdx.reshape((len(cpts),-1,free_param.size))
        if self.trajectories._parameters['use_chains']:
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
        
        DdX = sparse.csr_matrix(DdX)
        DXU = sparse.csr_matrix(DXU)
    
    def get_guess(self, interpolate=None):
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
        '''
        if not self.trajectories._old_splines:
            if interpolate is None:
                free_coeffs_all = np.hstack(self.trajectories.indep_coeffs.values())
                guess = 0.1 * np.ones(free_coeffs_all.size)
            else:
                pass
        else:
            guess = np.empty(0)
            
            # now we compute a new guess for every free coefficient of every new (finer) spline
            # by interpolating the corresponding old (coarser) spline
            for k, v in sorted(self.trajectories.indep_coeffs.items(), key = lambda (k, v): k):
                if (self.trajectories.splines[k].type == 'x'):
                    logging.debug("Get new guess for spline {}".format(k))
                    
                    s_new = self.trajectories.splines[k]
                    s_old = self.trajectories._old_splines[k]

                    df0 = s_old.df(self.trajectories._a)
                    dfn = s_old.df(self.trajectories._b)

                    free_coeffs_guess = s_new.interpolate(s_old.f, m0=df0, mn=dfn)
                    guess = np.hstack((guess, free_coeffs_guess))
                    
                    if 0 and s_new._use_std_approach:
                        # compute values 
                        values = [s_old.f(t) for t in s_new.nodes]
                        
                        # create vector of step sizes
                        h = np.array([s_new.nodes[k+1] - s_new.nodes[k] for k in xrange(s_new.nodes.size-1)])
                        
                        # create diagonals for the coefficient matrix of the equation system
                        l = np.array([h[k+1] / (h[k] + h[k+1]) for k in xrange(s_new.nodes.size-2)])
                        d = 2.0*np.ones(s_new.nodes.size-2)
                        u = np.array([h[k] / (h[k] + h[k+1]) for k in xrange(s_new.nodes.size-2)])
                        
                        # right hand site of the equation system
                        r = np.array([(3.0/h[k])*l[k]*(values[k+1] - values[k]) + (3.0/h[k+1])*u[k]*(values[k+2]-values[k+1])\
                                      for k in xrange(s_new.nodes.size-2)])
                        
                        # add conditions for unique solution
                        
                        # boundary derivatives
                        l = np.hstack([l, 0.0, 0.0])
                        d = np.hstack([1.0, d, 1.0])
                        u = np.hstack([0.0, 0.0, u])
                        
                        m0 = s_old.df(s_old.a)
                        mn = s_old.df(s_old.b)
                        r = np.hstack([m0, r, mn])
                        
                        data = [l,d,u]
                        offsets = [-1, 0, 1]
                        
                        # create tridiagonal coefficient matrix
                        D = sparse.dia_matrix((data, offsets), shape=(s_new.n+1, s_new.n+1))
                        
                        # solve the equation system
                        sol = sparse.linalg.spsolve(D.tocsr(),r)
                        
                        # calculate the coefficients
                        coeffs = np.zeros((s_new.n, 4))
                        
                        # compute the coefficients of the interpolant
                        for i in xrange(s_new.n):
                            coeffs[i, :] = [-2.0/h[i]**3 * (values[i+1]-values[i]) + 1.0/h[i]**2 * (sol[i]+sol[i+1]),
                                            3.0/h[i]**2 * (values[i+1]-values[i]) - 1.0/h[i] * (2*sol[i]+sol[i+1]),
                                            sol[i],
                                            values[i]]
                        
                        # get the indices of the free coefficients
                        coeff_name_split_str = [c.name.split('_')[-2:] for c in s_new._indep_coeffs]
                        free_coeff_indices = [(int(s[0]), int(s[1])) for s in coeff_name_split_str]
                        
                        free_coeffs_guess = np.array([coeffs[i] for i in free_coeff_indices])
                        guess = np.hstack((guess, free_coeffs_guess))
                    elif 0:
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
                    guess = np.hstack((guess, self.trajectories._old_splines[k]._indep_coeffs))
        
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
        solver = Solver(F=G, DF=DG, x0=self.guess, tol=self._parameters['tol'],
                        maxIt=self._parameters['sol_steps'], method=self._parameters['method'])
        
        # solve the equation system
        self.sol = solver.solve()
        
        return self.sol

    def save(self):

        save = dict()

        # parameters
        save['parameters'] = self._parameters

        # vector field and jacobian
        save['f'] = self._f
        save['Df'] = self._Df

        # guess
        save['guess'] = self.guess
        
        # solution
        save['sol'] = self.sol

        return save

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

def compare_trajectories(trajectories, free_coeffs, plot=True, embed=True):
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    
    tt = np.linspace(trajectories._a, trajectories._b, 1000)
    
    xt_old = np.zeros((1000,len(trajectories._x_sym)))
    x_old_nodes = []
    for i, x in enumerate(trajectories._x_sym):
        s = trajectories._old_splines[x]
        xt_old[:,i] = [s.f(t) for t in tt]
        x_old_nodes.append([s.f(t) for t in s.nodes])
    x_old_nodes = np.array(x_old_nodes).T
    old_nodes = s.nodes

    splines_bak = trajectories._splines.copy()
    trajectories.set_coeffs(free_coeffs)
    
    xt_new = np.zeros((1000,len(trajectories._x_sym)))
    x_new_nodes = []
    for i, x in enumerate(trajectories._x_sym):
        s = trajectories._splines[x]
        xt_new[:,i] = [s.f(t) for t in tt]
        x_new_nodes.append([s.f(t) for t in s.nodes])
    x_new_nodes = np.array(x_new_nodes).T
    new_nodes = s.nodes
    
    print np.abs(xt_old-xt_new).max()
    
    if plot or np.abs(xt_old-xt_new).max() > 0.5:
        plt.plot(tt, xt_old,'r--')
        plt.plot(old_nodes, x_old_nodes, 'ro', markersize=8, fillstyle='full')
        
        plt.plot(tt, xt_new, 'g')
        plt.plot(new_nodes, x_new_nodes, 'go', markersize=5, fillstyle='full')
        
        plt.show()
    
    if embed or np.abs(xt_old-xt_new).max() > 0.5:
        IPS()
    
    trajectories._splines = splines_bak
    for s in trajectories._splines.values():
        s._prov_flag = True

def _get_derivation_order(fnc):
    '''
    Returns derivation order of function according to place in integrator chain.
    '''

    from .splines import Spline
    
    if fnc.im_func == Spline.f.im_func:
        return 0
    elif fnc.im_func == Spline.df.im_func:
        return 1
    elif fnc.im_func == Spline.ddf.im_func:
        return 2
    elif fnc.im_func == Spline.dddf.im_func:
        return 3
    else:
        raise ValueError()

def _build_sol_from_free_coeffs(splines):
    '''
    Concatenates the values of the independent coeffs
    of all splines in given dict to build pseudo solution.
    '''

    sol = np.empty(0)
    for k, v in sorted(splines.items(), key = lambda (k, v): k):
        assert not v._prov_flag
        sol = np.hstack([sol, v._indep_coeffs])

    return sol
