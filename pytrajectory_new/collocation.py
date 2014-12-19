# IMPORTS
import numpy as np
import sympy as sp
from scipy import sparse
import logging

from solver import Solver

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
    def __init__(self, CtrlSys):
        self.sys = CtrlSys
    
    
    def build(self):
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

        logging.debug("  Building Equation System")
        
        # make functions local
        x_fnc = self.sys.trajectories._x_fnc
        dx_fnc = self.sys.trajectories._dx_fnc
        u_fnc = self.sys.trajectories._u_fnc

        # make symbols local
        x_sym = self.sys.x_sym
        u_sym = self.sys.u_sym

        a = self.sys.a
        b = self.sys.b
        delta = self.sys.mparam['delta']

        # now we generate the collocation points
        if self.sys.mparam['colltype'] == 'equidistant':
            # get equidistant collocation points
            cpts = np.linspace(a,b,(self.sys.mparam['sx']*delta+1),endpoint=True)
        elif self.sys.mparam['colltype'] == 'chebychev':
            # determine rank of chebychev polynomial
            # of which to calculate zero points
            nc = int(self.sys.mparam['sx']*delta - 1)

            # calculate zero points of chebychev polynomial --> in [-1,1]
            cheb_cpts = [np.cos( (2.0*i+1)/(2*(nc+1)) * np.pi) for i in xrange(nc)]
            cheb_cpts.sort()

            # transfer chebychev nodes from [-1,1] to our interval [a,b]
            a = self.sys.a
            b = self.sys.b
            chpts = [a + (b-a)/2.0 * (chp + 1) for chp in cheb_cpts]

            # add left and right borders
            cpts = np.hstack((a, chpts, b))
        else:
            logging.warning('Unknown type of collocation points.')
            logging.warning('--> will use equidistant points!')
            cpts = np.linspace(a,b,(self.sys.mparam['sx']*delta+1),endpoint=True)

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
        for k, v in sorted(self.sys.trajectories.indep_coeffs.items(), key=lambda (k, v): k.name):
            # increase j by the number of indep coeffs on which it depends
            j += len(v)
            indic[k] = (i, j)
            i = j

        # iterate over all quantities including inputs
        for sq in x_sym+u_sym:
            for ic in self.sys.chains:
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
        f = self.sys.ff_sym(x_sym,u_sym)
        Df_mat = sp.Matrix(f).jacobian(x_sym+u_sym)
        Df = sp.lambdify(x_sym+u_sym, Df_mat, modules='numpy')

        # the following would be created with every call to self.DG but it is possible to
        # only do it once. So we do it here to speed things up.

        # here we compute the jacobian matrix of the derivatives of the system state functions
        # (as they depend on the free parameters in a linear fashion its just the above matrix Mdx)
        DdX = Mdx.reshape((len(cpts),-1,len(self.c_list)))[:,self.sys.eqind,:]
        DdX = np.vstack(DdX)

        # here we compute the jacobian matrix of the system/input functions as they also depend on
        # the free parameters
        DXU = []
        x_len = len(self.sys.x_sym)
        u_len = len(self.sys.u_sym)
        xu_len = x_len + u_len

        for i in xrange(len(cpts)):
            DXU.append(np.vstack(( Mx[x_len*i:x_len*(i+1)], Mu[u_len*i:u_len*(i+1)] )))
        DXU_old = DXU
        DXU = np.vstack(DXU)
        
        if self.sys.mparam['use_sparse']:
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
        ff = self.sys.ff
        
        eqind = self.sys.eqind
        
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
            
            return G.flatten()
        
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
            if self.sys.mparam['use_chains']:
                DF_csr = [row for idx,row in enumerate(DF_csr.toarray()[:]) if idx%x_len in eqind]
                DF_csr = sparse.csr_matrix(DF_csr)
            
            DG = DF_csr - DdX
            
            return DG
        
        # return the callable functions
        return G, DG
    
    
    def get_guess(self):
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

        if (self.sys.nIt == 1):
            self.c_list = np.empty(0)

            for k, v in sorted(self.sys.trajectories.indep_coeffs.items(), key = lambda (k, v): k.name):
                self.c_list = np.hstack((self.c_list, v))
            guess = 0.1*np.ones(len(self.c_list))
        else:
            # make splines local
            old_splines = self.sys.trajectories._old_splines
            new_splines = self.sys.trajectories._splines

            guess = np.empty(0)
            self.c_list = np.empty(0)

            # get new guess for every independent variable
            for k, v in sorted(self.sys.trajectories.coeffs_sol.items(), key = lambda (k, v): k.name):
                self.c_list = np.hstack((self.c_list, self.sys.trajectories.indep_coeffs[k]))

                if (new_splines[k].type == 'x'):
                    logging.debug("Get new guess for spline %s"%k.name)

                    # how many unknown coefficients does the new spline have
                    nn = len(self.sys.trajectories.indep_coeffs[k])

                    # and this will be the points to evaluate the old spline in
                    #   but we don't want to use the borders because they got
                    #   the boundary values already
                    #gpts = np.linspace(self.a,self.b,(nn+1),endpoint = False)[1:]
                    #gpts = np.linspace(self.a,self.b,(nn+1),endpoint = True)
                    gpts = np.linspace(self.sys.a,self.sys.b,nn,endpoint = True)

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
                    guess = np.hstack((guess, self.sys.trajectories.coeffs_sol[k]))

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
        
        #from scipy import optimize as op
        #scipy_sol = op.root(fun=self.G, x0=self.guess, method='lm', jac=False)
        #IPS()
        
        return self.sol
    
        