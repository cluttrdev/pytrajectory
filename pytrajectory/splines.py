import numpy as np
import sympy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

from log import logging

# DEBUG
from IPython import embed as IPS

# NEW
from auxiliary import BetweenDict


class Spline(object):
    '''
    This class provides a representation of a cubic spline function.
    
    It simultaneously provides access to the spline function itself as well as to its derivatives
    up to the 3rd order. Furthermore it has its own method to ensure the steadiness and smoothness 
    conditions of its polynomial parts in the joining points.
    
    For more information see: :ref:`candidate_functions`
    
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
    
    bv : dict
        Boundary values the spline function and/or its derivatives should satisfy.
    
    nodes_type : str
        The type of the spline nodes (equidistant).
    
    '''
    
    def __init__(self, a=0.0, b=1.0, n=5, bv={}, nodes_type='equidistant', tag=''):
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
        self._boundary_values = bv
        
        # create array of symbolic coefficients
        self._coeffs = sp.symarray('c'+tag, (self.n, 4))
        
        # calculate nodes of the spline
        self.nodes = get_spline_nodes(self.a, self.b, self.n+1, nodes_type)
        self._nodes_type = nodes_type
        
        # create an dictionary with
        #   key: the intervals defined by the spline nodes
        #   values: the corresponding polynomial spline part
        self._nodes_dict = BetweenDict()
        for i in xrange(self.n):
            self._nodes_dict[(self.nodes[i], self.nodes[i+1])] = i
        self._nodes_dict[(self.nodes[self.n], np.inf)] = self.n-1
        
        # size of each polynomial part
        self._h = (self.b - self.a) / float(self.n)
        
        # the polynomial spline parts
        #   key: spline part
        #   value: corresponding polynomial
        self._S = dict()
        for i in xrange(self.n):
            # create polynomials, e.g. for cubic spline:
            #   S_i(t)= c_i_3*t^3 + c_i_2*t^2 + c_i_1*t + c_i_0
            self._S[i] = np.poly1d(self._coeffs[i])
        
        # initialise array for provisionally evaluation of the spline
        # if there are no values for its free parameters
        # 
        # they show how the spline coefficients depend on the free coefficients
        self._dep_array = None  #np.array([])
        self._dep_array_abs = None  #np.array([])
        
        # steady flag is True if smoothness and boundary conditions are solved
        # --> make_steady()
        self._steady_flag = False
        
        # provisionally flag is True as long as there are no numerical values
        # for the free parameters of the spline
        # --> set_coefficients()
        self._prov_flag = True
        
        # the free parameters of the spline
        self._indep_coeffs = None #np.array([])
    
    def __getitem__(self, key):
        return self._S[key]
    
    def _eval(self, t, d=0):
        '''
        Returns the value of the spline's `d`-th derivative at `t`.
        
        Parameters
        ----------
        
        t : float
            The point at which to evaluate the spline `d`-th derivative
        
        d : int
            The derivation order
        '''
        
        # get polynomial part where t is in
        #i = self._nodes_dict[t]
        i = int(np.floor(t * self.n / self.b))
        if i == self.n: i -= 1
        
        #if i != min(int(np.floor(t * self.n / self.b)), self.n-1):
        #if i != self._nodes_dict[t]:
        #    from IPython import embed as IPS
        #    IPS()
        
        #return self._S[i](t - self.nodes[i])
        return self._S[i].deriv(d)(t - (i+1)*self._h)
    
    def f(self, t):
        '''This is just a wrapper to evaluate the spline itself.'''
        if not self._prov_flag:
            return self._eval(t, d=0)
        else:
            return self.get_dependence_vectors(t, d=0)

    def df(self, t):
        '''This is just a wrapper to evaluate the spline's 1st derivative.'''
        if not self._prov_flag:
            return self._eval(t, d=1)
        else:
            return self.get_dependence_vectors(t, d=1)
        
    def ddf(self, t):
        '''This is just a wrapper to evaluate the spline's 2nd derivative.'''
        if not self._prov_flag:
            return self._eval(t, d=2)
        else:
            return self.get_dependence_vectors(t, d=2)
        
    def dddf(self, t):
        '''This is just a wrapper to evaluate the spline's 3rd derivative.'''
        if not self._prov_flag:
            return self._eval(t, d=3)
        else:
            return self.get_dependence_vectors(t, d=3)
    
    @property
    def boundary_values(self):
        return self._boundary_values
    
    @boundary_values.setter
    def boundary_values(self, value):
        self._boundary_values = value
    
    def make_steady(self):
        make_steady(S=self)
    
    def differentiate(self, d=1, new_tag=''):
        '''
        Returns the `d`-th derivative of this spline function object.
        
        Parameters
        ----------
        
        d : int
            The derivation order.
        '''
        return differentiate(self, d, new_tag)
    
    def get_dependence_vectors(self, points, d=0):
        '''
        This method yields a provisionally evaluation of the spline 
        while there are no numerical values for its free parameters.
        
        It returns a two vectors which reflect the dependence of the 
        spline's or its `d`-th derivative's coefficients on its free 
        parameters (independent coefficients).
        
        Parameters
        ----------
        
        points : float
            The points to evaluate the provisionally spline at.
        
        d : int
            The derivation order.
        '''
        
        #IPS()
        if np.size(points) > 1:
            raise NotImplementedError()
        t = points
        
        # determine the spline part to evaluate
        #i = self._nodes_dict[t]
        i = int(np.floor(t * self.n / self.b))
        if i == self.n: i -= 1
        
        #if i != min(int(np.floor(t * self.n / self.b)), self.n-1):
        #if i != self._nodes_dict[t]:
        #    from IPython import embed as IPS
        #    IPS()
        
        #t -= self.nodes[i]
        t -= (i+1) * self._h
        
        # Calculate vector to for multiplication with coefficient matrix w.r.t. the derivation order
        if d == 0:
            tt = np.array([t*t*t, t*t, t, 1.0])
        elif d == 1:
            tt = np.array([3.0*t*t, 2.0*t, 1.0, 0.0])
        elif d == 2:
            tt = np.array([6.0*t, 2.0, 0.0, 0.0])
        elif d == 3:
            tt = np.array([6.0, 0.0, 0.0, 0.0])
        
        dep_vec = np.dot(tt, self._dep_array[i])
        dep_vec_abs = np.dot(tt, self._dep_array_abs[i])
        
        return dep_vec, dep_vec_abs
    
    
    def set_coefficients(self, free_coeffs=None, coeffs=None):
        '''
        This function is used to set up numerical values either for all the spline's coefficients
        or its independent ones.
        
        Parameters
        ----------
        
        free_coeffs : numpy.ndarray
            Array with numerical values for the free coefficients of the spline.
        
        coeffs : numpy.ndarray
            Array with coefficients of the polynomial spline parts.
        
        '''
        
        # deside what to do
        if coeffs is None  and free_coeffs is None:
            # nothing to do
            pass
        
        elif coeffs is not None and free_coeffs is None:
            # set all the coefficients for the spline's polynomial parts
            # 
            # first a little check
            if not (self.n == coeffs.shape[0]):
                logging.error('Dimension mismatch in number of spline parts ({}) and \
                            rows in coefficients array ({})'.format(self.n, coeffs.shape[0]))
                raise ValueError('Dimension mismatch in number of spline parts ({}) and \
                            rows in coefficients array ({})'.format(self.n, coeffs.shape[0]))
            elif not (coeffs.shape[1] == 4):
                logging.error('Dimension mismatch in number of polynomial coefficients (4) and \
                            columns in coefficients array ({})'.format(coeffs.shape[1]))
            # elif not (self._indep_coeffs.size == coeffs.shape[1]):
            #     logging.error('Dimension mismatch in number of free coefficients ({}) and \
            #                 columns in coefficients array ({})'.format(self._indep_coeffs.size, coeffs.shape[1]))
            #     raise ValueError
            
            # set coefficients
            self._coeffs = coeffs
            
            # update polynomial parts
            for k in xrange(self.n):
                self._S[k] = np.poly1d(self._coeffs[k])
        
        elif coeffs is None and free_coeffs is not None:
            # a little check
            if not (self._indep_coeffs.size == free_coeffs.size):
                logging.error('Got {} values for the {} independent coefficients.'\
                                .format(free_coeffs.size, self._indep_coeffs.size))
                raise ValueError('Got {} values for the {} independent coefficients.'\
                                .format(free_coeffs.size, self._indep_coeffs.size))
            
            # set the numerical values
            self._indep_coeffs = free_coeffs
            
            # update the spline coefficients and polynomial parts
            for k in xrange(self.n):
                coeffs_k = self._dep_array[k].dot(free_coeffs) + self._dep_array_abs[k]
                self._coeffs[k] = coeffs_k
                self._S[k] = np.poly1d(coeffs_k)
        else:
            # not sure...
            logging.error('Not sure what to do, please either pass `coeffs` or `free_coeffs`.')
            raise TypeError('Not sure what to do, please either pass `coeffs` or `free_coeffs`.')
        
        # now we have numerical values for the coefficients so we can set this to False
        self._prov_flag = False
    
    
    def interpolate(self, fnc=None, points=None):
        '''
        Determines the spline's coefficients such that it interpolates
        a given function `fnc` or discrete `points`.
        
        '''
        
        if not points:
            points = self.nodes
        
        assert self._steady_flag
    
        # check if the spline should interpolate a function or given points
        if callable(fnc):
            # how many independent coefficients does the spline have
            coeffs_size = self._indep_coeffs.size
        
            # generate points to evaluate the function at
            # (function and spline interpolant should be equal in these)
            #points = np.linspace(S.a, S.b, coeffs_size, endpoint=False)
            nodes = np.linspace(self.a, self.b, coeffs_size, endpoint=True)
        
            # evaluate the function
            fnc_t = np.array([fnc(t) for t in nodes])
        
            dep_vecs = [self.get_dependence_vectors(t) for t in nodes]
            S_dep_mat = np.array([vec[0] for vec in dep_vecs])
            S_dep_mat_abs = np.array([vec[1] for vec in dep_vecs])
        
            # solve the equation system
            #free_coeffs = np.linalg.solve(S_dep_mat, fnc_t - S_dep_mat_abs)
            free_coeffs = np.linalg.lstsq(S_dep_mat, fnc_t - S_dep_mat_abs)[0]
        else:
            # get nodes and values
            points = np.array(points)
            
            assert points.ndim == 2
            
            shape = points.shape
            if shape[0] == 2:
                nodes = points[0,:]
                values = points[1,:]
            elif shape[1] == 2:
                nodes = points[:,0]
                values = points[:,1]
        
            assert self.a <= nodes[0] and nodes[-1] <= self.b
            assert nodes.size == values.size == self._indep_coeffs.size
            
            # get dependence matrices of the spline's coefficients
            dep_vecs = [self.get_dependence_vectors(t) for t in nodes]
            S_dep_mat = np.array([vec[0] for vec in dep_vecs])
            S_dep_mat_abs = np.array([vec[1] for vec in dep_vecs])
        
            # solve the equation system
            #free_coeffs = np.linalg.solve(S_dep_mat, fnc_t - S_dep_mat_abs)
            free_coeffs = np.linalg.lstsq(S_dep_mat, values - S_dep_mat_abs)[0]
        
        # set solution for the free coefficients
        self.set_coefficients(free_coeffs=free_coeffs)
    
    def plot(self, show=True, ret_array=False):
        '''
        Plots the spline function or returns an array with its values at
        some points of the spline interval.
        
        Parameters
        ----------
        
        show : bool
            Whethter to plot the spline's curve or not.
        
        ret_array : bool
            Wheter to return an array with values of the spline at points
            of the interval.
            
        '''
        
        if not show and not ret_array:
            # nothing to do here...
            return
        elif self._prov_flag:
            # spline cannot be plotted, because there are no numeric
            # values for its polynomial coefficients
            logging.error("There are no numeric values for the spline's\
                            polynomial coefficients.")
            return
        
        # create array of values
        tt = np.linspace(self.a, self.b, 1000, endpoint=True)
        St = [self.f(t) for t in tt]
        
        if show:
            try:
                import matplotlib.pyplot as plt
                plt.plot(tt,St)
                plt.show()
            except ImportError:
                logging.error('Could not import matplotlib for plotting the curve.')
        
        if ret_array:
            return St
    

def get_spline_nodes(a=0.0, b=1.0, n=10, nodes_type='equidistant'):
    '''
    Generates :math:`n` spline nodes in the interval :math:`[a,b]`
    of given type.
    
    Parameters
    ----------
    
    a : float
        Lower border of the considered interval.
    
    b : float
        Upper border of the considered interval.
    
    n : int
        Number of nodes to generate.
    
    nodes_type : str
        How to generate the nodes.
    '''
    
    if nodes_type == 'equidistant':
        nodes = np.linspace(a, b, n, endpoint=True)
    else:
        raise NotImplementedError()
    
    return nodes
    
    
def differentiate(spline_fnc):
    '''
    Returns the derivative of a callable spline function.
    
    Parameters
    ----------
    
    spline_fnc : callable
        The spline function to derivate.
    
    '''
    # `im_func` is the function's id
    # `im_self` is the object of which `func` is the method
    if spline_fnc.im_func == Spline.f.im_func:
        return spline_fnc.im_self.df
    elif spline_fnc.im_func == Spline.df.im_func:
        return spline_fnc.im_self.ddf
    elif spline_fnc.im_func == Spline.ddf.im_func:
        return spline_fnc.im_self.dddf
    else:
        raise NotImplementedError()
    
def make_steady(S):
    '''
    This method sets up and solves equations that satisfy boundary conditions and
    ensure steadiness and smoothness conditions of the spline `S` in every joining point.
    
    Please see the documentation for more details: :ref:`candidate_functions`
    
    Parameters
    ----------
    
    S : Spline
        The spline function object for which to solve smoothness and boundary conditions.
    '''
    
    # This should be yet untouched
    if S._steady_flag:
        logging.warning('Spline already has been made steady.')
        return
    
    # get spline coefficients and interval size
    coeffs = S._coeffs
    h = S._h

    # nu represents degree of boundary conditions
    nu = -1
    for k, v in S._boundary_values.items():
        if all(item is not None for item in v):
            nu += 1
    
    # now we determine the free parameters of the spline function
    if nu == -1:
        a = np.hstack([coeffs[:,0], coeffs[0,1:]])
    elif nu == 0:
        a = np.hstack([coeffs[:,0], coeffs[0,2]])
    elif nu == 1:
        a = coeffs[:-1,0]
    elif nu == 2:
        a = coeffs[:-3,0]
    
    # `b` is, what is not in `a`
    coeffs_set = set(coeffs.ravel())
    a_set = set(a)
    b_set = coeffs_set - a_set
    
    # transfer b_set to ordered list
    b = sorted(list(b_set), key = lambda c: c.name)
    #b = np.array(sorted(list(b_set), key = lambda c: c.name))
    
    # now we build the matrix for the equation system
    # that ensures the smoothness conditions
    
    # get matrix dimensions --> (3.21) & (3.22)
    N1 = 3 * (S.n - 1) + 2 * (nu + 1)
    N2 = 4 * S.n
    
    # get matrix and right hand site of the equation system
    # that ensures smoothness and compliance with the boundary values
    M, r = get_smoothness_matrix(S, N1, N2)
    
    # get A and B matrix such that
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
    A = sparse.csc_matrix(A)
    B = sparse.csc_matrix(B)
    r = sparse.csc_matrix(r)
    
    tmp1 = spsolve(B,r)
    tmp2 = spsolve(B,-A)
    
    if sparse.issparse(tmp1):
        tmp1 = tmp1.toarray()
    if sparse.issparse(tmp2):
        tmp2 = tmp2.toarray()
    
    dep_array = np.zeros((coeffs.shape[0], coeffs.shape[1], a.size))
    dep_array_abs = np.zeros_like(coeffs, dtype=float)
    
    for i,bb in enumerate(b):
        tmp = bb.name.split('_')[-2:]
        j = int(tmp[0])
        k = int(tmp[1])

        dep_array[j,k,:] = tmp2[i]
        dep_array_abs[j,k] = tmp1[i]

    tmp3 = np.eye(len(a))
    for i,aa in enumerate(a):
        tmp = aa.name.split('_')[-2:]
        j = int(tmp[0])
        k = int(tmp[1])

        dep_array[j,k,:] = tmp3[i]
    
    S._dep_array = dep_array
    S._dep_array_abs = dep_array_abs
    
    # a is vector of independent spline coeffs (free parameters)
    S._indep_coeffs = a
    
    # now we are done and this can be set to True
    S._steady_flag = True

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
    
    n = S.n
    h = S._h
    
    # initialise the matrix and the right hand site
    M = sparse.lil_matrix((N1,N2))
    r = sparse.lil_matrix((N1,1))
    
    # build block band matrix M for smoothness conditions 
    # in every joining point
    block = np.array([[0.0, 0.0, 0.0, 1.0,   h**3, -h**2,  h, -1.0],
                      [0.0, 0.0, 1.0, 0.0, -3*h**2, 2*h, -1.0, 0.0],
                      [0.0, 2.0, 0.0, 0.0,   6*h,  -2.0,  0.0, 0.0]])
    
    for k in xrange(n-1):
        M[3*k:3*(k+1),4*k:4*(k+2)] = block
    
    # add equations for boundary values
    
    # for the spline function itself
    if S._boundary_values.has_key(0):
        if S._boundary_values[0][0] is not None:
            M[3*(n-1),0:4] = np.array([-h**3, h**2, -h, 1.0])
            r[3*(n-1)] = S._boundary_values[0][0]
        if S._boundary_values[0][1] is not None:
            M[3*(n-1)+1,-4:] = np.array([0.0, 0.0, 0.0, 1.0])
            r[3*(n-1)+1] = S._boundary_values[0][1]
    # for its 1st derivative
    if S._boundary_values.has_key(1):
        if S._boundary_values[1][0] is not None:
            M[3*(n-1)+2,0:4] = np.array([3*h**2, -2*h, 1.0, 0.0])
            r[3*(n-1)+2] = S._boundary_values[1][0]
        if S._boundary_values[1][1] is not None:
            M[3*(n-1)+3,-4:] = np.array([0.0, 0.0, 1.0, 0.0])
            r[3*(n-1)+3] = S._boundary_values[1][1]
    # and for its 2nd derivative
    if S._boundary_values.has_key(2):
        if S._boundary_values[2][0] is not None:
            M[3*(n-1)+4,0:4] = np.array([-6*h, 2.0, 0.0, 0.0])
            r[3*(n-1)+4] = S._boundary_values[2][0]
        if S._boundary_values[2][1] is not None:
            M[3*(n-1)+5,-4:] = np.array([0.0, 2.0, 0.0, 0.0])
            r[3*(n-1)+5] = S._boundary_values[2][1]
    
    return M, r


