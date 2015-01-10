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
    
    nodes_type : str
        The type of the spline nodes (equidistant/chebychev).
    
    steady : bool
        Whether or not to call :meth:`make_steady()` when instanciated.
    
    '''
    
    def __init__(self, a=0.0, b=1.0, n=10, bc={}, poly_order=-1, nodes_type='equidistant', steady=False, tag='', use_std_def=False):
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
        self._deriv_order = 0   #deriv_order
        
        # create array of symbolic coefficients
        self._coeffs = sp.symarray('c'+tag, (self.n, self._poly_order+1))
        
        # calculate nodes of the spline
        self.nodes = get_spline_nodes(self.a, self.b, self.n+1, nodes_type)
        self._nodes_type = nodes_type
        
        ##############
        # NEW
        self._nodes_dict = BetweenDict()
        for i in xrange(self.n):
            self._nodes_dict[(self.nodes[i], self.nodes[i+1])] = i
        self._nodes_dict[(self.nodes[self.n], np.inf)] = self.n-1
        ##############
        
        # vector of step sizes
        self._h = np.array([self.nodes[i+1] - self.nodes[i] for i in xrange(self.n)])
        
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
        # they show how the spline coefficients depend on its free
        # coefficients
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
        self._indep_coeffs = np.array([])
        
        # whether to use the standard spline definition (True)
        # or the one used in Oliver Schnabel's project thesis (False)
        self._use_std_def = use_std_def
        
        if steady:
            self.make_steady()
    
    
    def __getitem__(self, key):
        return self._S[key]
    
    def __call__(self, t):
        # get polynomial part where t is in
        i = self._nodes_dict[t]
        
        if self._use_std_def:
            return self._S[i](t - self.nodes[i])
        else:
            return self._S[i](t - self.nodes[i+1])
        
    
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
        return self._deriv_order
    
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
        Returns the `d`-th derivative of this spline function object.
        
        Parameters
        ----------
        
        d : int
            The derivation order.
        '''
        return differentiate(self, d, new_tag)
    
    def get_dependence_vectors(self, points):
        '''
        This method yields a provisionally evaluation of the spline while there are no numerical 
        values for its free parameters.
        It returns a two vectors which reflect the dependence of the spline coefficients
        on its free parameters (independent coefficients).
        
        
        Parameters
        ----------
        
        points : float / array_like
            The points to evaluate the provisionally spline at.
        
        '''
        
        #IPS()
        if np.size(points) > 1:
            raise NotImplementedError()
        t = points
        
        # determine the spline part to evaluate
        i = self._nodes_dict[t]
        
        if self._use_std_def:
            t = t - self.nodes[i]
        else:
            t = t - self.nodes[i+1]
        
        tt = [t*t*t, t*t, t, 1.0][-(self._poly_order + 1):]
        
        M0 = self._dep_array[i]
        m0 = self._dep_array_abs[i]
        
        dep_vec = np.dot(tt,M0)
        dep_vec_abs = np.dot(tt,m0)
        
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
            # set all coefficients of the spline's polynomial parts
            # 
            # first a little check
            if not (self.n == coeffs.shape[0]):
                logging.error('Dimension mismatch in number of spline parts ({}) and \
                            rows in coefficients array ({})'.format(self.n, coeffs.shape[0]))
                raise ValueError
            elif not (self._poly_order + 1 == coeffs.shape[1]):
                logging.error('Dimension mismatch in number of polynomial coefficients ({}) and \
                            columns in coefficients array ({})'.format(self._poly_order + 1, coeffs.shape[1]))
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
                                .format(free_coeffs.size,self._indep_coeffs.size))
                raise ValueError
            
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
            raise RuntimeError
        
        # now we have numerical values for the coefficients so we can set this to False
        self._prov_flag = False
    
    
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
        St = [self(t) for t in tt]
        
        if show:
            try:
                import matplotlib.pyplot as plt
                plt.plot(tt,St)
                plt.show()
            except ImportError:
                logging.error('Could not import matplotlib for plotting the curve.')
        
        if ret_array:
            return St
    

class ConstantSpline(Spline):
    '''
    This class provides a spline object with piecewise constant polynomials.
    '''
    def __init__(self, a=0.0, b=1.0, n=10, bc=dict(), nodes_type='equidistant', steady=False, tag='', use_std_def=False):
        Spline.__init__(self, a=a, b=b, n=n, tag=tag, bc=bc, poly_order=0, nodes_type=nodes_type, steady=steady, use_std_def=use_std_def)

class LinearSpline(Spline):
    '''
    This class provides a spline object with piecewise linear polynomials.
    '''
    def __init__(self, a=0.0, b=1.0, n=10, bc=dict(), nodes_type='equidistant', steady=False, tag='', use_std_def=False):
        Spline.__init__(self, a=a, b=b, n=n, tag=tag, bc=bc, poly_order=1, nodes_type=nodes_type, steady=steady, use_std_def=use_std_def)


class QuadraticSpline(Spline):
    '''
    This class provides a spline object with piecewise quadratic polynomials.
    '''
    def __init__(self, a=0.0, b=1.0, n=10, bc=dict(), nodes_type='equidistant', steady=False, tag='', use_std_def=False):
        Spline.__init__(self, a=a, b=b, n=n, tag=tag, bc=bc, poly_order=2, nodes_type=nodes_type, steady=steady, use_std_def=use_std_def)


class CubicSpline(Spline):
    '''
    This class provides a spline object with piecewise cubic polynomials.
    '''
    def __init__(self, a=0.0, b=1.0, n=10, bc=dict(), nodes_type='equidistant', steady=False, tag='', use_std_def=False):
        Spline.__init__(self, a=a, b=b, n=n, tag=tag, bc=bc, poly_order=3, nodes_type=nodes_type, steady=steady, use_std_def=use_std_def)
            

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
    
    nodes_type : str (equidistant/chebychev)
        How to generate the nodes.
    '''
    
    if nodes_type == 'equidistant':
        nodes = np.linspace(a, b, n, endpoint=True)
    elif nodes_type == 'chebychev':
        # determine rank of chebychev polynomial
        # of which to calculate zero points
        nc = int(n) - 2

        # calculate zero points of chebychev polynomial --> in [-1,1]
        cheb_cpts = [np.cos( (2.0*i+1)/(2*(nc+1)) * np.pi) for i in xrange(nc)]
        cheb_cpts.sort()

        # transfer chebychev nodes from [-1,1] to our interval [a,b]
        chpts = [a + (b-a)/2.0 * (chp + 1) for chp in cheb_cpts]

        # add left and right borders
        nodes = np.hstack((a, chpts, b))
    else:
        raise NotImplementedError()
    
    return nodes
    
    
def differentiate(S, d=1, new_tag=''):
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
    
    # if d == 0:
    #     #return S
    #     pass
    if d > 3:
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
            dS = CubicSpline(a=a, b=b, n=n, tag=new_tag, bc=None, steady=False)
        elif po == 2:
            dS = QuadraticSpline(a=a, b=b, n=n, tag=new_tag, bc=None, steady=False)
        elif po == 1:
            dS = LinearSpline(a=a, b=b, n=n, tag=new_tag, bc=None, steady=False)
        elif po == 0:
            dS = ConstantSpline(a=a, b=b, n=n, tag=new_tag, bc=None, steady=False)
        
        dS._deriv_order = do
        dS._steady_flag = True
        
        # determine the coefficients of the new derivative
        coeffs = S._coeffs.copy()[:,:(po + 1)]
        
        # get the matrices for provisionally evaluation of the spline
        dep_array = S._dep_array.copy()[:,:(po + 1)]
        dep_array_abs = S._dep_array_abs.copy()[:,:(po + 1)]
        
        # now consider factors that result from differentiation
        if S.is_cubic():
            if d == 1:
                coeffs[:,0] *= 3
                coeffs[:,1] *= 2
                
                dep_array[:,0] *= 3
                dep_array[:,1] *= 2
                dep_array_abs[:,0] *= 3
                dep_array_abs[:,1] *= 2
            elif d == 2:
                coeffs[:,0] *= 6
                coeffs[:,1] *= 2
                
                dep_array[:,0] *= 6
                dep_array[:,1] *= 2
                dep_array_abs[:,0] *= 6
                dep_array_abs[:,1] *= 2
            elif d == 3:
                coeffs[:,0] *= 6
                
                dep_array[:0] *= 6
                dep_array_abs[:,0] *= 6
        elif S.is_quadratic():
            if d == 1:
                coeffs[:,0] *= 2
                
                dep_array[:,0] *= 2
                dep_array_abs[:,0] *= 2
        
        dS._coeffs = coeffs
        dS._dep_array = dep_array
        dS._dep_array_abs = dep_array_abs
        
        # they have their independent coefficients in common
        dS._indep_coeffs = S._indep_coeffs
        
        dS.nodes = S.nodes#.copy()
        dS._nodes_dict = S._nodes_dict#.copy()
        
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
    if S._steady_flag:
        logging.warning('Spline already has been made steady.')
        return
    
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
        dep_array = np.zeros((coeffs.shape[0],coeffs.shape[1],a.size))
        dep_array_abs = np.zeros_like(coeffs, dtype=float)
        
        
        eye = np.eye(len(a))
        
        if nu == -1:
            for i in xrange(S.n):
                dep_array[i,0,:] = eye[i]
        elif nu == 0:
            for i in xrange(1,S.n-1):
                dep_array[i,0,:] = eye[i-1]
            
            dep_array[0,0,:] = np.zeros(len(a))
            dep_array[-1,0,:] = np.zeros(len(a))
            dep_array_abs[0,0] = S._bc[0][0]
            dep_array_abs[-1,0] = S._bc[0][1]
        
        S._dep_array = dep_array
        S._dep_array_abs = dep_array_abs
    
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
    
    dep_array = np.zeros((coeffs.shape[0],coeffs.shape[1],a.size))
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
    
    return S


def determine_indep_coeffs(S, nu=-1):
    '''
    Determines the vector of independent coefficients w.r.t. the 
    degree of boundary conditions `nu`.
    
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
            a = np.hstack([coeffs[:,0], coeffs[0,2]])
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
    
    if S.is_constant():
        raise NotImplementedError()
    
    if S._use_std_def:
        for k in xrange(n-1):
            if S.is_cubic():
                block = np.array([[  h[k]**3, h[k]**2,  h[k], 1.0, 0.0, 0.0, 0.0, -1.0],
                                  [3*h[k]**2,  2*h[k],  1.0,  0.0, 0.0, 0.0, -1.0, 0.0],
                                  [  6*h[k],    2.0,    0.0,  0.0, 0.0, -2.0, 0.0, 0.0]])
            elif S.is_quadratic():
                block = np.array([[h[k]**2, h[k], 1.0, 0.0, 0.0, -1.0],
                                  [ 2*h[k],  1.0, 0.0, 0.0, -1.0, 0.0]])
            elif S.is_linear():
                block = np.array([[h[k], 1.0, 0.0, -1.0]])
        
            M[po*k:po*(k+1),(po+1)*k:(po+1)*(k+2)] = block
    
        # add equations for boundary conditions
        if S._bc.has_key(0) and not any(item is None for item in S._bc[0]):
            M[po*(n-1),0:(po+1)] =   np.array([   0.0,       0.0,     0.0,   1.0])
            M[po*(n-1)+1,-(po+1):] = np.array([h[n-1]**3, h[n-1]**2, h[n-1], 1.0])
            r[po*(n-1)] = S._bc[0][0]
            r[po*(n-1)+1] = S._bc[0][1]
        if S._bc.has_key(1) and not any(item is None for item in S._bc[1]):
            M[po*(n-1)+2,0:(po+1)] = np.array([    0.0,       0.0,    1.0, 0.0])
            M[po*(n-1)+3,-(po+1):] = np.array([3*h[n-1]**2, 2*h[n-1], 1.0, 0.0])
            r[po*(n-1)+2] = S._bc[1][0]
            r[po*(n-1)+3] = S._bc[1][1]
        if S._bc.has_key(2) and not any(item is None for item in S._bc[2]):
            M[po*(n-1)+4,0:(po+1)] = np.array([  0.0,    2.0, 0.0, 0.0])
            M[po*(n-1)+5,-(po+1):] = np.array([6*h[n-1], 2.0, 0.0, 0.0])
            r[po*(n-1)+4] = S._bc[2][0]
            r[po*(n-1)+5] = S._bc[2][1]
    else:
        for k in xrange(n-1):
            if S.is_cubic():
                block = np.array([[0.0, 0.0, 0.0, 1.0,   h[k]**3, -h[k]**2, h[k], -1.0],
                                  [0.0, 0.0, 1.0, 0.0, -3*h[k]**2,  2*h[k], -1.0,  0.0],
                                  [0.0, 2.0, 0.0, 0.0,   6*h[k],    -2.0,    0.0,  0.0]])
            elif S.is_quadratic():
                block = np.array([[0.0, 0.0, 1.0, -h[k]**2, h[k], -1.0],
                                  [0.0, 1.0, 0.0,  2*h[k],  -1.0, 0.0]])
            elif S.is_linear():
                block = np.array([[0.0, 1.0, h[k], -1.0]])
        
            M[po*k:po*(k+1),(po+1)*k:(po+1)*(k+2)] = block
        
        # add equations for boundary conditions
        if S._bc.has_key(0) and not any(item is None for item in S._bc[0]):
            M[po*(n-1),0:(po+1)] =   np.array([-h[n-1]**3, h[n-1]**2, -h[n-1], 1.0])
            M[po*(n-1)+1,-(po+1):] = np.array([    0.0,        0.0,     0.0,   1.0])
            r[po*(n-1)] = S._bc[0][0]
            r[po*(n-1)+1] = S._bc[0][1]
        if S._bc.has_key(1) and not any(item is None for item in S._bc[1]):
            M[po*(n-1)+2,0:(po+1)] = np.array([3*h[n-1]**2, -2*h[n-1], 1.0, 0.0])
            M[po*(n-1)+3,-(po+1):] = np.array([    0.0,        0.0,    1.0, 0.0])
            r[po*(n-1)+2] = S._bc[1][0]
            r[po*(n-1)+3] = S._bc[1][1]
        if S._bc.has_key(2) and not any(item is None for item in S._bc[2]):
            M[po*(n-1)+4,0:(po+1)] = np.array([-6*h[n-1], 2.0, 0.0, 0.0])
            M[po*(n-1)+5,-(po+1):] = np.array([   0.0,    2.0, 0.0, 0.0])
            r[po*(n-1)+4] = S._bc[2][0]
            r[po*(n-1)+5] = S._bc[2][1]
    
    return M, r


def interpolate(S=None, fnc=None, points=None, n_nodes=100, nodes_type='equidistant', spline_order=3):
    '''
    Interpolates a given function or dicrete points using a
    spline function object which will be created if not passed.
    
    Parameters
    ----------
    
    S : Spline
        The spline function object used to interpolate.
    
    fnc : callable
        The function that should be interpolated.
    
    points : array_like
        One or two dimensional array containing interval borders, nodes or 
        2d-points that should be interpolated.
    
    n_nodes : int
        The number of nodes that the interpolating spline should have
        (if the interpolant `S` is not given).
    
    nodes_type : str
        The type of the spline nodes (if the interpolant `S` is not given).
    
    spline_order : int
        The polynomial order of the spline parts (if the interpolant `S` is not given).
    
    '''
    
    if not spline_order in {0,1,3}:
        raise NotImplementedError()
    
    # first check passed arguments
    if points.ndim == 1:
        # `points` is assumed to contain either interval borders or interpolation nodes
        # so `fnc` has to be given and callable
        assert callable(fnc)
        
        if len(points) == 2:
            # `points` is assumed to contain interval borders so the interpolation nodes
            # have to be generated
            a, b = points
            nodes = np.linspace(a, b, n_nodes, endpoint=True)
        elif len(points) > 2:
            # `points` is assumed to contain interpolation nodes
            a = points[0]
            b = points[-1]
            nodes = points.copy()
        else:
            raise ValueError('Input argument `points` must at least containt interval borders.')
        
        # get values at the nodes
        values = np.array([fnc(node) for node in nodes]).ravel()
        
        # make sure `fnc` has the right dimension
        if not values.ndim == 1:
            raise ValueError('Can only interpolate 1-dimensional function, not {}-dimensional.'.format(vaues.ndim))
    elif points.ndim == 2:
        # `points` is assumed to contain the interpolation nodes and values
        # so `fnc` should not be callable
        assert not callable(fnc)
        
        # get nodes and values
        shape = points.shape
        if shape[0] >= shape[1]:
            nodes = points[:,0]
            values = points[:,1]
        else:
            nodes = points[0,:]
            values = points[1,:]
    
    # create spline function object if not given
    if not S:
        if spline_order == 0:
            S = ConstantSpline(a=nodes[0], b=nodes[-1], n=nodes.size, nodes_type=nodes_type)
        elif spline_order == 1:
            S = LinearSpline(a=nodes[0], b=nodes[-1], n=nodes.size - 1, nodes_type=nodes_type)
        elif spline_order == 2:
            S = QuadraticSpline(a=nodes[0], b=nodes[-1], n=nodes.size - 1, nodes_type=nodes_type)
        elif spline_order == 3:
            S = CubicSpline(a=nodes[0], b=nodes[-1], n=nodes.size - 1, nodes_type=nodes_type)
    else:
        # check attributes of the given spline function
        if S.is_constant():
            assert S.n == nodes.size
        else:
            assert S.n == nodes.size - 1
        
        assert S.a == nodes[0] and S.b == nodes[-1]
    
    # make sure that the spline has not already been made steady (and smooth)
    if S._steady_flag:
        logging.error('Spline should not have been made steady for interpolation.')
        raise NotImplementedError('Spline should not have been made steady for interpolation.')
    
    if S.is_cubic():
        # create vector of step sizes
        h = np.array([nodes[k+1] - nodes[k] for k in xrange(nodes.size-1)])
        
        # create diagonals for the coefficient matrix of the equation system
        l = np.array([h[k+1] / (h[k] + h[k+1]) for k in xrange(nodes.size-2)])
        d = 2.0*np.ones(nodes.size-2)
        u = np.array([h[k] / (h[k] + h[k+1]) for k in xrange(nodes.size-2)])
        
        # right hand site of the equation system
        r = np.array([(3.0/h[k])*l[k]*(values[k+1] - values[k]) + (3.0/h[k+1])*u[k]*(values[k+2]-values[k+1])\
                        for k in xrange(nodes.size-2)])
        
        # add conditions for unique solution
        # 
        # natural spline
        l = np.hstack([l, 1.0, 0.0])
        d = np.hstack([2.0, d, 2.0])
        u = np.hstack([0.0, 1.0, u])
        r = np.hstack([(3.0/h[0])*(values[1]-values[0]), r, (3.0/h[-1])*(values[-1]-values[-2])])
        
        data = [l,d,u]
        offsets = [-1, 0, 1]
        
        # create tridiagonal coefficient matrix
        D = sparse.dia_matrix((data, offsets), shape=(S.n+1,S.n+1))
        
        # solve the equation system
        try:
            sol = sparse.linalg.spsolve(D.tocsr(),r)
        except:
            print "SOL ERROR"
            IPS()
        
        # calculate the coefficients
        coeffs = np.zeros((S.n, 4))
        
        for i in xrange(S.n):
            coeffs[i, :] = [-2.0/h[i]**3 * (values[i+1]-values[i]) + 1.0/h[i]**2 * (sol[i]+sol[i+1]),
                         3.0/h[i]**2 * (values[i+1]-values[i]) - 1.0/h[i] * (2*sol[i]+sol[i+1]),
                         sol[i],
                         values[i]]
    elif S.is_linear():
        coeffs = np.zeros((S.n, 2))
        
        for i in xrange(S.n):
            coeffs[i,0] = (values[i+1] - values[i]) / (nodes[i+1] - nodes[i])
        coeffs[:,1] = values[:-1]
    
    elif S.is_constant():
        # to get a constant interpolant we use the nearest neighbor interpolation
        # 
        # therefore we have to recalculate the spline nodes
        # as the midpoints of the current interval parts
        # 
        # to keep the interval borders we have to add a node
        # (and therefore a spline part in the initialisation above)
        new_nodes = np.zeros(nodes.size + 1)
        new_nodes[0] = nodes[0]
        new_nodes[-1] = nodes[-1]
        for i in xrange(1, nodes.size):
            new_nodes[i] = 0.5 * (nodes[i-1] + nodes[i])
        
        # change spline nodes manually
        S.nodes = new_nodes
        S._nodes_dict = BetweenDict()
        for i in xrange(S.n):
            key = (new_nodes[i], new_nodes[i+1])
            S._nodes_dict[key] = i
        S._nodes_dict[(new_nodes[-1], np.inf)] = S.n - 1
        
        coeffs = np.zeros((S.n, 1))
        coeffs[:,0] = values[:]
    
    # set solution
    S.set_coefficients(coeffs=coeffs)
    
    return S
    
        


if __name__=='__main__':
    CS = CubicSpline()
    
    if 0:
        S = interpolate(fnc=np.sin, points=np.linspace(0,2*np.pi,10,endpoint=True), spline_order=3)
        tt = np.linspace(0,2*np.pi,1000)
        St = [S(t) for t in tt]
        sint = [np.sin(t) for t in tt]
        
        #idx = [(int(p[-3]),int(p[-1])) for p in str(CS._indep_coeffs)[1:-1].split(' ')]
    IPS()
