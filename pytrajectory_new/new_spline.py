##############################################################
# NEW: HIGHLY EXPERIMENTAL AND PRE-ALPHA

# DON'T USE THIS BY NOW
##############################################################
import numpy as np
import sympy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

import logging

# DEBUG
from IPython import embed as IPS



class Spline():
    '''
    This class provides a base spline function object.
    
    Parameters
    ----------
    
    a : float
        Left border of the spline interval.
    
    b : float
        Right border of the spline interval.
    
    n : int
        Number of polynomial parts the spline will be devided up into.
    
    tag : str
        The 'name' of the spline object.
    
    bc : dict
        Boundary values the spline function and/or its derivatives should satisfy.
    
    poly_order : int
        The order of the polynomial spline parts.
    
    deriv_order : int
        If not 0 this spline is the :attr:`deriv_order`-th derivative of another spline.
    
    steady : bool
        Whether or not to call :meth:`makesteady()` when instanciated.
    
    '''
    
    def __init__(self, a=0.0, b=1.0, n=10, tag='s', bc=None, poly_order=-1, deriv_order=0, steady=False):
        # interval boundaries
        assert a < b
        self.a = a
        self.b = b
        
        # number of polynomial parts
        self.n = int(n)
        
        # 'name' of the spline
        self.tag = tag
        
        # dictionary with boundary values
        #   key: order of the spline's derivative to which the values belong
        #   values: the boundary values the derivative should satisfy
        self._bc = bc
        
        # type of the polynomial parts
        self._order = poly_order
        
        if self._order == -1:
            self._type = None
        elif self._order == 0:
            self._type == 'constant'
        elif self._order == 1:
            self._type = 'linear'
        elif self._order == 2:
            self._type = 'quadratic'
        elif self._order == 3:
            self._type = 'cubic'
        
        # is this spline object are derivative of another one?
        # if not -> 0
        # else   -> the derivation order
        self._deriv = deriv_order
        
        # size of the polynomial parts for equidistant nodes
        self._h = (float(self.b)-float(self.a))/float(self.n)
        
        # create array of symbolic coefficients
        self._coeffs = sp.symarray('c'+tag, (self.n, self._order+1))
        
        # reverse the order of the symbols in every column
        #for i in xrange(int((self._order+1)/2)):
        #    tmp = self._coeffs[:,i].copy()
        #    self._coeffs[:,i] = self._coeffs[:,self._order-i]
        #    self._coeffs[:,self._order-i] = tmp
        
        # the polynomial spline parts
        #   key: spline part
        #   value: corresponding polynomial
        self._S = dict()
        for i in xrange(self.n):
            # create polynomials, e.g. for cubic spline:
            #   S_i(t)= c_i_3*t^3 + c_i_2*t^2 + c_i_1*t + c_i_0
            self._S[i] = np.poly1d(self._coeffs[i])
        
        # create matrices for provisionally evaluation of the spline
        # if there are no values for its free parameters
        self._prov_S = dict()   #np.ones_like(self._coeffs)
        self._prov_S_abs = dict()   #np.zeros((self.n,self._order+1))
        
        # steady flag is True if smoothness and boundary conditions are solved
        # --> makesteady()
        self._steady_flag = False
        
        # provisionally flag is True as long as there are no numerical values
        # for the free parameters of the spline
        # --> set_coeffs()
        self._prov_flag = True
        
        # calculate joining points of the spline
        self._jpts = np.linspace(self.a, self.b, self.n+1)
        
        # the free parameters of the spline
        self._free_coeffs = None
        
        if (steady):
            self.make_steady()
    
    
    def __getitem__(self, i):
        return self._S[i]
    
    def __call__(self, t):
        raise NotImplementedError
    
    def _prov_evalf(self, tt, i):
        '''
        This function yields a provisionally evaluation of the spline while there are no numerical 
        values for its free parameters.
        It returns a two vectors which reflect the dependence of the spline coefficients
        on its free parameters (independent coefficients).
        
        
        Parameters
        ----------
        
        tt : tuple
            The vector with the powers of the polynomial spline part evaluated at a some point.
        
        i : int
            The polynomial spline part to evaluate.
        
        '''
        
        M0 = np.array([m for m in self._prov_S[i]], dtype=float)
        m0 = self._prov_S_abs[i]
        
        return np.dot(tt,M0), np.dot(tt,m0)
    
    
    def _evalf(self, t):
        '''
        Returns the value of the spline at :attr:`t`.
        
        Parameters
        ----------
        
        t : float
            The point to evaluate the spline at
        '''
        
        # get polynomial part where t is in
        i = int(np.floor(t*self.n/(self.b)))
        # if `t` is equal to the right border, which is the last node, there is no
        # corresponding spline part so we use the one before
        if (i == self.n): i-= 1
        
        t -= (i+1)*self._h
        #t -= self._jpts[i]
        
        return self._S[i](t)
    
    
    def is_constant(self):
        return self._type == 'constant'
    
    def is_linear(self):
        return self._type == 'linear'
    
    def is_quadratic(self):
        return self._type == 'quadratic'
    
    def is_cubic(self):
        return self._type == 'cubic'
    
    def is_derivative(self):
        '''
        Returns `0` if this spline object is not a derivative of another one, else the derivation order.
        '''
        return self._deriv
    
    def boundary_values(self, d=None, bv=None):
        '''
        Set the boundary values that the :attr:`d`-th derivative of this spline object should satisfy
        or, if no arguments are given, return the current dictionary of boundary values.
        
        Parameters
        ----------
        
        d : int
            The derivation order.
        
        bv : tuple
            The boundary values the :attr:`d`-th derivative should satisfy.
        '''
        if (d != None) and (bv != None):
            self._bc[d] = bv
        else:
            if self._bc:
                return self._bc.copy()
            else:
                return {}
    
    def make_steady(self):
        self = make_steady(self)
    
    
    def derive(self, d=1):
        '''
        Returns the d-th derivative of this spline function object.
        
        Parameters
        ----------
        
        d : int
            The derivation order.
        '''
        return derive_spline(self, d)
    
    def set_coeffs(self, coeffs=None, free_coeffs=None):
        '''
        This function is used to set up numerical values either for all the spline's coefficients
        or its independent ones.
        
        Parameters
        ----------
        
        coeffs : numpy.ndarray
            Array with coefficients of the polynomial spline parts.
        
        free_coeffs : numpy.ndarray
            Array with numerical values for the free coefficients of the spline.
        '''
        
        # deside what to do
        if not (coeffs and free_coeffs):
            # nothing to do
            pass
        
        elif coeffs and not free_coeffs:
            # set all coefficients of the spline's polynomial parts
            # 
            # first a little check
            if not (self.n == coeffs.shape[0]):
                logging.error('Dimension mismatch in number of spline parts ({}) and \
                            rows in coefficients array ({})'.format(self.n, coeffs.shape[0]))
                raise ValueError
            elif not (self._free_coeffs.size == coeffs.shape[1]):
                logging.error('Dimension mismatch in number of free coefficients ({}) and \
                            columns in coefficients array ({})'.format(self._free_coeffs.size, coeffs.shape[1]))
                raise ValueError
            
            # set coefficients
            self._coeffs = coeffs
            
            # update polynomial parts
            for k in xrange(self.n):
                self._S[k] = np.poly1d(self._coeffs[k])
        
        elif not coeffs and free_coeffs:
            # a little check
            if not (self._free_coeffs.size == free_coeffs.size):
                logging.error('Got less values ({}) for the independent coefficients ({})'\
                                .format(free_coeffs.size,self._free_coeffs.size))
                raise ValueError
            
            # set the numerical values
            self._free_coeffs = free_coeffs
            
            # update the spline coefficients and polynomial parts
            for k in xrange(self.n):
                self._coeffs[k] = [row.dot(self._free_coeffs) + _abs \
                                    for row, _abs in zip(self._prov_S[k], self._prov_S_abs)]
                
                self._S[k] = np.poly1d(self._coeffs[k])
        else:
            # not sure...
            logging.error('Not sure what to do, please either pass `coeffs` or `free_coeffs`.')
            raise RuntimeError
        
        # now we have numerical values for the coefficients so we can set this to False
        self._prov_flag = False
    


def derive_spline(S, d=1, new_tag=''):
    '''
    Returns the :attr:`d`-th derivative of the given spline function object :attr:`S`.
    
    Parameters
    ----------
    
    S : Spline
        The spline function object to derivate.
    
    d : int
        The derivation order.
    
    new_tag : str
        New name for the derivative.
    '''
    
    if not S._steady_flag:
        logging.error('The spline to derive has to be made steady')
        return None
    
    if not new_tag:
        new_tag = 'd' + S.tag
    
    if d == 0:
        return S
    elif d > 3:
        raise Exception
    else:
        # first, get things that the spline and its derivative have in common
        a = S.a
        b = S.b
        n = S.n
        
        # calculate new polynomial order
        po = S._order - d
        
        # get and increase derivation order flag
        do = S._deriv + d
        
        # create new spline object
        dS = Spline(a=a, b=b, n=n, tag=new_tag, bc=None, poly_order=po, deriv_order=do, steady=False)
        
        dS._steady_flag = True
        
        # determine the coefficients of the new derivative
        coeffs = S._coeffs.copy()[:,:-d]
        
        if d == 1:
            coeffs[:,0] *= 3
            coeffs[:,1] *= 2
        elif d == 2:
            coeffs[:,0] *= 6
            coeffs[:,1] *= 2
        elif d == 3:
            coeffs[:,0] *= 6
        
        dS._coeffs = coeffs
        
        # they have their independent coefficients in common
        dS._free_coeffs = S._free_coeffs
        
        # get the matrices for provisionally evaluation of the spline
        for k, v in S._prov_S.iteritems():
            dS._prov_S[k] = v[:-d]
        
        for k, v in S._prov_S_abs.iteritems():
            dS._prov_S_abs[k] = v[:-d]
        
        # set polynomial parts
        for k in xrange(dS.n):
            dS._S[k] = np.poly1d(dS._coeffs[k])
            
        return dS
        

def make_steady(S):
    '''
    This method sets up and solves equations that satisfy boundary conditions and
    ensure steadiness and smoothness conditions of the spline :attr:`S` in every joining point.
    
    Please see the documentation for more details: :ref:`candidate_functions`
    
    Parameters
    ----------
    
    S : Spline
        The spline function object to solve smoothness and boundary conditions for.
    '''
    
    # This should be yet untouched
    assert S._steady_flag == False

    coeffs = S._coeffs
    h = S._h

    # nu represents degree of boundary conditions
    try:
        nu = len(S._bc.keys()) - 1
    except:
        nu = -1
    
    # now we determine the free parameters of the spline function
    if (nu == -1):
        a = coeffs[:,0]
        a = np.hstack((a,coeffs[0,1:]))
    if (nu == 0):
        a = coeffs[:,0]
        a = np.hstack((a,coeffs[0,3]))
    if (nu == 1):
        a = coeffs[:-1,0]
    if (nu == 2):
        a = coeffs[:-3,0]

    # b is what is not in a
    coeffs_set = set(coeffs.flatten())
    a_set = set(a)
    b_set = coeffs_set - a_set
    
    # transfer b_set to ordered list
    b = sorted(list(b_set), key = lambda c: c.name)
    #b = np.array(sorted(list(b_set), key = lambda c: c.name))
    
    # now we build the matrix for the equation system
    # that ensures the smoothness conditions

    # get matrix dimensions --> (3.21) & (3.22)
    N2 = 4*S.n
    N1 = 3*(S.n-1) + 2*(nu+1)
    
    # the following may cause MemoryError
    # TODO: (optionally) introduce sparse matrix already here
    #M = np.zeros((N1,N2))
    #r = np.zeros(N1)
    M = sparse.lil_matrix((N1,N2))
    r = sparse.lil_matrix((N1,1))
    
    # build block band matrix M for smoothness in every joining point
    #   derivatives from order 0 to 2
    block = np.array([[0.0, 0.0, 0.0, 1.0,   h**3, -h**2,  h,  -1.0],
                       [0.0, 0.0, 1.0, 0.0, -3*h**2, 2*h, -1.0, 0.0],
                       [0.0, 2.0, 0.0, 0.0,    6*h,  -2.0,  0.0, 0.0]])
    
    for i in xrange(S.n-1):
        M[3*i:3*(i+1),4*i:4*(i+2)] = block
    
    # add equations for boundary conditions
    if S._bc.has_key(0):
        M[3*(S.n-1),0:4] = np.array([-h**3, h**2, -h, 1.0])
        M[3*(S.n-1)+1,-4:] = np.array([0.0, 0.0, 0.0, 1.0])
        r[3*(S.n-1)] = S._bc[0][0]
        r[3*(S.n-1)+1] = S._bc[0][1]
    if S._bc.has_key(1):
        M[3*(S.n-1)+2,0:4] = np.array([3*h**2, -2*h, 1.0, 0.0])
        M[3*(S.n-1)+3,-4:] = np.array([0.0, 0.0, 1.0, 0.0])
        r[3*(S.n-1)+2] = S._bc[1][0]
        r[3*(S.n-1)+3] = S._bc[1][1]
    if S._bc.has_key(2):
        M[3*(S.n-1)+4,0:4] = np.array([-6*h, 2.0, 0.0, 0.0])
        M[3*(S.n-1)+5,-4:] = np.array([0.0, 2.0, 0.0, 0.0])
        r[3*(S.n-1)+4] = S._bc[2][0]
        r[3*(S.n-1)+5] = S._bc[2][1]

    # get A and B matrix --> see docu
    #
    #       M*c = r
    # A*a + B*b = r
    #         b = B^(-1)*(r-A*a)
    #
    # we need B^(-1)*r [absolute part -> tmp1] and B^(-1)*A [coefficients of a -> tmp2]

    #a_mat = np.zeros((N2,N2-N1))
    #b_mat = np.zeros((N2,N1))
    a_mat = sparse.lil_matrix((N2,N2-N1))
    b_mat = sparse.lil_matrix((N2,N1))
    
    for i,aa in enumerate(a):
        tmp = aa.name.split('_')[-2:]
        j = int(tmp[0])
        k = int(tmp[1])
        a_mat[4*j+k,i] = 1

    for i,bb in enumerate(b):
        tmp = bb.name.split('_')[-2:]
        j = int(tmp[0])
        k = int(tmp[1])
        b_mat[4*j+k,i] = 1
    
    M = sparse.csr_matrix(M)
    a_mat = sparse.csr_matrix(a_mat)
    b_mat = sparse.csr_matrix(b_mat)
    
    A = M.dot(a_mat)
    B = M.dot(b_mat)
    
    # do the inversion
    #tmp1 = np.array(np.linalg.solve(B,r.T),dtype=np.float)
    #tmp2 = np.array(np.linalg.solve(B,-A),dtype=np.float)
    A = sparse.csc_matrix(A)
    B = sparse.csc_matrix(B)
    r = sparse.csc_matrix(r)
    tmp1 = spsolve(B,r)
    tmp2 = spsolve(B,-A)
    
    if sparse.issparse(tmp1):
        tmp1 = tmp1.toarray()
    if sparse.issparse(tmp2):
        tmp2 = tmp2.toarray()
    
    tmp_coeffs = np.zeros_like(S._coeffs, dtype=None)
    tmp_coeffs_abs = np.zeros((S.n,S._order+1))

    for i,bb in enumerate(b):
        tmp = bb.name.split('_')[-2:]
        j = int(tmp[0])
        k = int(tmp[1])

        tmp_coeffs[(j,k)] = tmp2[i]
        tmp_coeffs_abs[(j,k)] = tmp1[i]

    tmp3 = np.eye(len(a))
    for i,aa in enumerate(a):
        tmp = aa.name.split('_')[-2:]
        j = int(tmp[0])
        k = int(tmp[1])

        tmp_coeffs[(j,k)] = tmp3[i]
    
    for i in xrange(S.n):
        S._prov_S[i] = tmp_coeffs[i]
        S._prov_S_abs[i] = tmp_coeffs_abs[i]

    # a is vector of independent spline coeffs (free parameters)
    S._free_coeffs = a
    
    # now we are done and this can be set to True
    S._steady_flag = True
    
    return S


class ConstantSpline(Spline):
    '''
    This class provides a spline object with piecewise constant polynomials.
    '''
    def __init__(self, a=0.0, b=1.0, n=10, tag='', bc=dict(), deriv_order=0, steady=False):
        Spline.__init__(self, a=a, b=b, n=n, tag=tag, bc=bc, poly_order=0, steady=steady)
    
    def __call__(self, t):
        if not self._prov_flag:
            return self._evalf(t)
        else:
            i = int(np.floor(t*self.n/(self.b)))
            if (i == self.n): i-= 1
            t -= (i+1)*self._h
            
            tt = [1.0]
            return self._prov_evalf(tt, i)


class LinearSpline(Spline):
    '''
    This class provides a spline object with piecewise linear polynomials.
    '''
    def __init__(self, a=0.0, b=1.0, n=10, tag='', bc=dict(), deriv_order=0, steady=False):
        Spline.__init__(self, a=a, b=b, n=n, tag=tag, bc=bc, poly_order=1, steady=steady)
    
    def __call__(self, t):
        if not self._prov_flag:
            return self._evalf(t)
        else:
            i = int(np.floor(t*self.n/(self.b)))
            if (i == self.n): i-= 1
            t -= (i+1)*self._h
            
            tt = [t, 1.0]
            return self._prov_evalf(tt, i)


class QuadraticSpline(Spline):
    '''
    This class provides a spline object with piecewise quadratic polynomials.
    '''
    def __init__(self, a=0.0, b=1.0, n=10, tag='', bc=dict(), deriv_order=0, steady=False):
        Spline.__init__(self, a=a, b=b, n=n, tag=tag, bc=bc, poly_order=2, steady=steady)
    
    def __call__(self, t):
        if not self._prov_flag:
            return self._evalf(t)
        else:
            i = int(np.floor(t*self.n/(self.b)))
            if (i == self.n): i-= 1
            t -= (i+1)*self._h
            
            tt = [t*t, t, 1.0]
            return self._prov_evalf(tt, i)


class CubicSpline(Spline):
    '''
    This class provides a spline object with piecewise cubic polynomials.
    '''
    def __init__(self, a=0.0, b=1.0, n=10, tag='', bc=dict(), deriv_order=0, steady=False):
        Spline.__init__(self, a=a, b=b, n=n, tag=tag, bc=bc, poly_order=3, steady=steady)
    
    def __call__(self, t):
        if not self._prov_flag:
            return self._evalf(t)
        else:
            i = int(np.floor(t*self.n/(self.b)))
            if (i == self.n): i-= 1
            t -= (i+1)*self._h
            
            tt = [t*t*t, t*t, t, 1.0]
            return self._prov_evalf(tt, i)
            


if __name__=='__main__':
    S = Spline()
    CS = CubicSpline()
    
#     x1, x2 = sp.symbols('x1, x2')
#     u1 = sp.Symbol('u1')
#
#     x_sym = [x1, x2]
#     u_sym = [u1]
#
#
#     xbc = {x1 : [0.0,1.0],
#             x2 : [0.0, 0.0]}
#     ubc = {u1 : [0.0, 0.0]}
#
#     a = 0.0
#     b = 2.0
#
#     initSplines(a=a, b=b, xbc=xbc, ubc=ubc, x_sym=x_sym, u_sym=u_sym)
#
#     if 0:
#         def f(x,u):
#             x1, x2 = x
#             u1, = u
#
#             ff = np.array([ x2,
#                             u1])
#             return ff
#         xa = [0.0, 0.0]
#         xb = [1.0, 0.0]
#         g = [0.0,0.0]
#         T = Trajectory(f, a=a, b=b, xa=xa, xb=xb, sx=5, su=5, kx=3, g=g, use_chains=False)
    
    
    IPS()
