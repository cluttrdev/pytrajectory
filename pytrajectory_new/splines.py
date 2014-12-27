import numpy as np
import sympy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

import logging

# DEBUG
from IPython import embed as IPS



class Spline(object):
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
        Whether or not to call :meth:`make_steady()` when instanciated.
    
    '''
    
    def __init__(self, a=0.0, b=1.0, n=5, tag='s', bc={}, poly_order=-1, deriv_order=0, steady=False):
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
        self._poly_order = poly_order
        
        if self._poly_order == -1:
            self._type = None
        elif self._poly_order == 0:
            self._type = 'constant'
        elif self._poly_order == 1:
            self._type = 'linear'
        elif self._poly_order == 2:
            self._type = 'quadratic'
        elif self._poly_order == 3:
            self._type = 'cubic'
        
        # is this spline object are derivative of another one?
        # if not -> 0
        # else   -> the derivation order
        self._deriv_order = deriv_order
        
        # size of the polynomial parts for equidistant nodes
        self._h = (float(self.b)-float(self.a))/float(self.n)
        
        # create array of symbolic coefficients
        self._coeffs = sp.symarray('c'+tag, (self.n, self._poly_order+1))
        
        # reverse the order of the symbols in every column
        #for i in xrange(int((self._poly_order+1)/2)):
        #    tmp = self._coeffs[:,i].copy()
        #    self._coeffs[:,i] = self._coeffs[:,self._poly_order-i]
        #    self._coeffs[:,self._poly_order-i] = tmp
        
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
        self._prov_S = np.zeros_like(self._coeffs)
        self._prov_S_abs = np.zeros((self.n,self._poly_order+1))
        
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
        self._indep_coeffs = None
        
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
        # Get the spline part where t is in
        #i = int(np.floor(t*self.n/(self.b)))
        #if (i == self.n): i-= 1
        
        #t -= (i+1)*self.h
        #t -= self.jpts[i]
        
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
        
        #t -= (i+1)*self._h
        #t -= self._jpts[i]
        
        return self._S[i](t-(i+1)*self._h)
    
    
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
    
    
    def derive(self, d=1, new_tag=''):
        '''
        Returns the d-th derivative of this spline function object.
        
        Parameters
        ----------
        
        d : int
            The derivation order.
        '''
        return derive_spline(self, d, new_tag)
    
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
        if coeffs is None  and (free_coeffs is None):
            # nothing to do
            pass
        
        elif coeffs is not None and free_coeffs is None:
            # set all coefficients of the spline's polynomial parts
            # 
            # first a little check
            if not (self.n == coeffs.shape[0]):
                logging.error('Dimension mismatch in number of spline parts ({}) and \
                            rows in coefficients array ({})'.format(self.n, coeffs.shape[0]))
                raise ValueError
            elif not (self._indep_coeffs.size == coeffs.shape[1]):
                logging.error('Dimension mismatch in number of free coefficients ({}) and \
                            columns in coefficients array ({})'.format(self._indep_coeffs.size, coeffs.shape[1]))
                raise ValueError
            
            # set coefficients
            self._coeffs = coeffs
            
            # update polynomial parts
            for k in xrange(self.n):
                self._S[k] = np.poly1d(self._coeffs[k])
        
        elif coeffs is None and free_coeffs is not None:
            # a little check
            if not (self._indep_coeffs.size == free_coeffs.size):
                logging.error('Got {} values for the {} independent coefficients.'\
                                .format(free_coeffs.size,self._indep_coeffs.size))
                raise ValueError
            
            # set the numerical values
            self._indep_coeffs = free_coeffs
            
            # update the spline coefficients and polynomial parts
            for k in xrange(self.n):
                try:
                    coeffs_k = [row.dot(free_coeffs) + _abs for row, _abs in zip(self._prov_S[k], self._prov_S_abs[k])]
                except:
                    IPS()
                
                self._coeffs[k] = coeffs_k
                self._S[k] = np.poly1d(coeffs_k)
        else:
            # not sure...
            logging.error('Not sure what to do, please either pass `coeffs` or `free_coeffs`.')
            raise RuntimeError
        
        # now we have numerical values for the coefficients so we can set this to False
        self._prov_flag = False
    

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
    def __init__(self, a=0.0, b=1.0, n=5, tag='', bc=dict(), deriv_order=0, steady=False):
        Spline.__init__(self, a=a, b=b, n=n, tag=tag, bc=bc, poly_order=3, steady=steady)
    
    def __call__(self, t):
        if not self._prov_flag:
            return self._evalf(t)
        else:
            i = int(np.floor(t*self.n/(self.b)))
            if (i == self.n): i-= 1
            t = t - (i+1)*self._h
            
            tt = [t*t*t, t*t, t, 1.0]
            return self._prov_evalf(tt, i)
            

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
    assert S._poly_order >= d
    
    if not S._steady_flag:
        logging.error('The spline to derive has to be made steady')
        return None
    
    if new_tag == '':
        new_tag = 'd' + S.tag
    
    if d == 0:
        #return S
        pass
    elif d > 3:
        raise ValueError('Invalid order of differentiation (%d), maximum is 3.'%(d))
    else:
        # first, get things that the spline and its derivative have in common
        a = S.a
        b = S.b
        n = S.n
        
        # calculate new polynomial order
        po = S._poly_order - d
        
        # get and increase derivation order flag
        do = S._deriv_order + d
        
        # create new spline object
        if po == 3:
            dS = CubicSpline(a=a, b=b, n=n, tag=new_tag, bc=None, deriv_order=do, steady=False)
        elif po == 2:
            dS = QuadraticSpline(a=a, b=b, n=n, tag=new_tag, bc=None, deriv_order=do, steady=False)
        elif po == 1:
            dS = LinearSpline(a=a, b=b, n=n, tag=new_tag, bc=None, deriv_order=do, steady=False)
        elif po == 0:
            dS = ConstantSpline(a=a, b=b, n=n, tag=new_tag, bc=None, deriv_order=do, steady=False)
        
        dS._steady_flag = True
        
        
        # determine the coefficients of the new derivative
        coeffs = S._coeffs.copy()[:,:(po + 1)]
        
        # get the matrices for provisionally evaluation of the spline
        prov_S = S._prov_S.copy()[:,:(po + 1)]
        prov_S_abs = S._prov_S_abs.copy()[:,:(po + 1)]
        
        # now consider factors that result from differentiation
        if S.is_cubic():
            if d == 1:
                coeffs[:,0] *= 3
                coeffs[:,1] *= 2
                
                prov_S[:,0] *= 3
                prov_S[:,1] *= 2
                prov_S_abs[:,0] *= 3
                prov_S_abs[:,1] *= 2
            elif d == 2:
                coeffs[:,0] *= 6
                coeffs[:,1] *= 2
                
                prov_S[:,0] *= 6
                prov_S[:,1] *= 2
                prov_S_abs[:,0] *= 6
                prov_S_abs[:,1] *= 2
            elif d == 3:
                coeffs[:,0] *= 6
                
                prov_S[:,0] *= 6
                prov_S_abs[:,0] *= 6
        elif S.is_quadratic():
            if d == 1:
                coeffs[:,0] *= 2
                
                prov_S[:,0] *= 2
                prov_S_abs[:,0] *= 2
        
        dS._coeffs = coeffs
        dS._prov_S = prov_S
        dS._prov_S_abs = prov_S_abs
        
        # they have their independent coefficients in common
        dS._indep_coeffs = S._indep_coeffs
        
        # set polynomial parts
        for k in xrange(dS.n):
            dS._S[k] = np.poly1d(dS._coeffs[k])
        
        if not S._prov_flag:
            dS._prov_flag = False
            
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
    po = S._poly_order

    # nu represents degree of boundary conditions
    nu = -1
    for k, v in S._bc.items():
        if all(item is not None for item in v):
            nu += 1
    
    # now we determine the free parameters of the spline function
    a = determine_indep_coeffs(S, nu)
    
    # b is what is not in a
    coeffs_set = set(coeffs.flatten())
    a_set = set(a)
    b_set = coeffs_set - a_set
    
    # transfer b_set to ordered list
    b = sorted(list(b_set), key = lambda c: c.name)
    #b = np.array(sorted(list(b_set), key = lambda c: c.name))
    
    # now we build the matrix for the equation system
    # that ensures the smoothness conditions
    
    # if the spline is piecewise constant it needs special treatment, 
    # becaus it can't be made 'steady' in this sense
    if S.is_constant():
        #raise NotImplementedError
        
        tmp_coeffs = np.zeros_like(S._coeffs, dtype=None)
        tmp_coeffs_abs = np.zeros((S.n,S._poly_order+1))
        
        eye = np.eye(len(a))
        
        if nu == -1:
            for i in xrange(S.n):
                tmp_coeffs[(i,0)] = eye[i]
        elif nu == 0:
            for i in xrange(1,S.n-1):
                tmp_coeffs[(i,0)] = eye[i-1]
            
            tmp_coeffs[(0,0)] = np.zeros(len(a))
            tmp_coeffs[(-1,0)] = np.zeros(len(a))
            tmp_coeffs_abs[(0,0)] = S._bc[0][0]
            tmp_coeffs_abs[(-1,0)] = S._bc[0][1]
        
        S._prov_S = tmp_coeffs
        S._prov_S_abs = tmp_coeffs_abs
    
        # a is vector of independent spline coeffs (free parameters)
        S._indep_coeffs = a
    
        # now we are done and this can be set to True
        S._steady_flag = True
    
        return S
        
    
    
    
    
    
    # get matrix dimensions --> (3.21) & (3.22)
    N1 = S._poly_order * (S.n - 1) + 2 * (nu + 1)
    N2 = (S._poly_order + 1) * S.n
    
    # the following may cause MemoryError
    # TODO: (optionally) introduce sparse matrix already here
    #M = np.zeros((N1,N2))
    #r = np.zeros(N1)
    M, r = get_smoothness_matrix(S, N1, N2)
    
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
        a_mat[(po+1)*j+k,i] = 1

    for i,bb in enumerate(b):
        tmp = bb.name.split('_')[-2:]
        j = int(tmp[0])
        k = int(tmp[1])
        b_mat[(po+1)*j+k,i] = 1
    
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
    tmp_coeffs_abs = np.zeros((S.n,S._poly_order+1))

    for i,bb in enumerate(b):
        tmp = bb.name.split('_')[-2:]
        j = int(tmp[0])
        k = int(tmp[1])

        tmp_coeffs[(j,k)] = tmp2[i]
        tmp_coeffs_abs[(j,k)] = tmp1[i]

    #tmp3 = sparse.identity(len(a)).tolil()
    tmp3 = np.eye(len(a))
    for i,aa in enumerate(a):
        tmp = aa.name.split('_')[-2:]
        j = int(tmp[0])
        k = int(tmp[1])

        tmp_coeffs[(j,k)] = tmp3[i]
    
    #for i in xrange(S.n):
    #    S._prov_S[i] = tmp_coeffs[i]
    #    S._prov_S_abs[i] = tmp_coeffs_abs[i]
    
    S._prov_S = tmp_coeffs
    S._prov_S_abs = tmp_coeffs_abs
    
    # a is vector of independent spline coeffs (free parameters)
    S._indep_coeffs = a
    
    # now we are done and this can be set to True
    S._steady_flag = True
    
    return S


def determine_indep_coeffs(S, nu=-1):
    '''
    Determines the vector of independent coefficients w.r.t. the 
    degree of boundary conditions `c`.
    
    Parameters
    ----------
    
    S : Spline
        The spline function object to determine the independent coefficients of.
    
    nu : int
        The degree of boundary conditions the spline has to satisfy.
    
    Returns
    -------
    
    array_like
        The independent coefficients of the spline function.
    '''
    
    assert nu <= S._poly_order
    
    coeffs = S._coeffs
    
    if S.is_cubic():
        if nu == -1:
            a = np.hstack([coeffs[:,0], coeffs[0,1:]])
        elif nu == 0:
            a = np.hstack([coeffs[:,0], coeffs[0,2]]) #-> [0,3] ?
        elif nu == 1:
            a = coeffs[:-1,0]
        elif nu == 2:
            a = coeffs[:-3,0]
    elif S.is_quadratic():
        if nu == -1:
            a = np.hstack([coeffs[:,0], coeffs[0,1:]])
        elif nu == 0:
            a = coeffs[:,0]
        elif nu == 1:
            a = coeffs[:-2,0]
    elif S.is_linear():
        if nu == -1:
            a = np.hstack([coeffs[:,0], coeffs[0,1]])
        elif nu == 0:
            a = coeffs[:-1,0]
    elif S.is_constant():
        if nu == -1:
            a = coeffs[:,0]
        elif nu == 0:
            a = coeffs[1:-1,0]
    
    return a


def get_smoothness_matrix(S, N1, N2):
    '''
    Returns the coefficient matrix and right hand site for the 
    equation system that ensures the spline's smoothness in its 
    joining points and its compliance with the boundary conditions.
    
    Parameters
    ----------
    
    S : Spline
        The spline function object to get the matrix for.
    
    N1 : int
        First dimension of the matrix.
    
    N2 : int
        Second dimension of the matrix.
    
    Returns
    -------
    
    array_like
        The coefficient matrix for the equation system.
    
    array_like
        The right hand site of the equation system.
    '''
    
    po = S._poly_order
    n = S.n
    h = S._h
    
    # initialise the matrix and the right hand site
    M = sparse.lil_matrix((N1,N2))
    r = sparse.lil_matrix((N1,1))
    
    # build block band matrix M for smoothness conditions 
    # in every joining point
    if S.is_cubic():
        block = np.array([[0.0, 0.0, 0.0, 1.0,   h**3, -h**2,  h,  -1.0],
                          [0.0, 0.0, 1.0, 0.0, -3*h**2, 2*h, -1.0, 0.0],
                          [0.0, 2.0, 0.0, 0.0,    6*h,  -2.0,  0.0, 0.0]])
    elif S.is_quadratic():
        block = np.array([[0.0, 0.0, 1.0, -h**2,  h,  -1.0],
                          [0.0, 1.0, 0.0,  2*h, -1.0, 0.0]])
    elif S.is_linear():
        block = np.array([[0.0, 1.0, h,  -1.0]])
    
    for i in xrange(n-1):
        M[po*i:po*(i+1),(po+1)*i:(po+1)*(i+2)] = block
    
    # add equations for boundary conditions
    if S._bc.has_key(0) and not any(item is None for item in S._bc[0]):
        M[po*(n-1),0:(po+1)] = -1.0 * block[0,-(po+1):] # np.array([-h**3, h**2, -h, 1.0])
        M[po*(n-1)+1,-(po+1):] = block[0,:(po+1)] # np.array([0.0, 0.0, 0.0, 1.0])
        r[po*(n-1)] = S._bc[0][0]
        r[po*(n-1)+1] = S._bc[0][1]
    if S._bc.has_key(1) and not any(item is None for item in S._bc[1]):
        M[po*(n-1)+2,0:(po+1)] = -1.0 * block[1,-(po+1):] # np.array([3*h**2, -2*h, 1.0, 0.0])
        M[po*(n-1)+3,-(po+1):] = block[1,:(po+1)] # np.array([0.0, 0.0, 1.0, 0.0])
        r[po*(n-1)+2] = S._bc[1][0]
        r[po*(n-1)+3] = S._bc[1][1]
    if S._bc.has_key(2) and not any(item is None for item in S._bc[2]):
        M[po*(n-1)+4,0:(po+1)] = -1.0 * block[2,-(po+1):] # np.array([-6*h, 2.0, 0.0, 0.0])
        M[po*(n-1)+5,-(po+1):] = block[2,:(po+1)] # np.array([0.0, 2.0, 0.0, 0.0])
        r[po*(n-1)+4] = S._bc[2][0]
        r[po*(n-1)+5] = S._bc[2][1]
    
    return M, r
    
        


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
