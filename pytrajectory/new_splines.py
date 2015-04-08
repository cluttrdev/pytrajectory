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
    
    bv : dict
        Boundary values the spline function and/or its derivatives should satisfy.
    
    nodes_type : str
        The type of the spline nodes (equidistant/chebychev).
    
    use_std_def : bool
        Whether to use the standard spline definition 
        or the one used in Oliver Schnabel's project thesis
    '''
    
    # it has no polynomial parts
    _poly_order = -1
    
    def __init__(self, a=0.0, b=1.0, n=0, bv={}, nodes_type='equidistant', tag='', use_std_def=False):
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
        
        # create an dictionary with
        #   key: the intervals defined by the spline nodes
        #   values: the corresponding polynomial spline part
        self._nodes_dict = BetweenDict()
        for i in xrange(self.n):
            self._nodes_dict[(self.nodes[i], self.nodes[i+1])] = i
        self._nodes_dict[(self.nodes[self.n], np.inf)] = self.n-1
        
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
        self._indep_coeffs = np.array([])
        
        # whether to use the standard spline definition (True)
        # or the one used in Oliver Schnabel's project thesis (False)
        self._use_std_def = use_std_def
    
    def __getitem__(self, key):
        return self._S[key]
    
    def __call__(self, t):
        # get polynomial part where t is in
        i = self._nodes_dict[t]
        
        if self._use_std_def:
            return self._S[i](t - self.nodes[i])
        else:
            return self._S[i](t - self.nodes[i+1])
    
    @property
    def is_constant(self):
        return self._poly_order == 0
    
    @property
    def is_linear(self):
        return self._poly_order == 1
    
    @property
    def is_quadratic(self):
        return self._poly_order == 2
    
    @property
    def is_cubic(self):
        return self._poly_order == 3
    
    @property
    def is_derivative(self):
        '''
        Returns `0` if this spline object is not a derivative of another one, else the derivation order.
        '''
        return self._deriv_order
    
    @property
    def boundary_values(self):
        return self._boundary_values
    
    @boundary_values.setter
    def boundary_values(self, value):
        self._boundary_values = value
    
    def make_steady(self):
        self = make_steady(self)
    
    def differentiate(self, d=1, new_tag=''):
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
        
        D = self._dep_array[i]
        Da = self._dep_array_abs[i]
        
        dep_vec = np.dot(tt,D)
        dep_vec_abs = np.dot(tt,Da)
        
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
                raise ValueError('Dimension mismatch in number of spline parts ({}) and \
                            rows in coefficients array ({})'.format(self.n, coeffs.shape[0]))
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
        Determines the spline coefficients such that it interpolates
        a given function `fnc` or discrete `points`.
        '''
        
        if not points:
            points = self.nodes
        
        interpolate(self, fnc=fnc, points=points)
    
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
    
    _poly_order = 0
    
    def __init__(self, a=0.0, b=1.0, n=10, bv=dict(), nodes_type='equidistant', tag='', use_std_def=False):
        Spline.__init__(self, a=a, b=b, n=n, tag=tag, bv=bv, nodes_type=nodes_type, use_std_def=use_std_def)

class LinearSpline(Spline):
    '''
    This class provides a spline object with piecewise linear polynomials.
    '''
    
    _poly_order = 1
    
    def __init__(self, a=0.0, b=1.0, n=10, bv=dict(), nodes_type='equidistant', tag='', use_std_def=False):
        Spline.__init__(self, a=a, b=b, n=n, tag=tag, bv=bv, nodes_type=nodes_type, use_std_def=use_std_def)


class QuadraticSpline(Spline):
    '''
    This class provides a spline object with piecewise quadratic polynomials.
    '''
    
    _poly_order = 2
    
    def __init__(self, a=0.0, b=1.0, n=10, bv=dict(), nodes_type='equidistant', tag='', use_std_def=False):
        Spline.__init__(self, a=a, b=b, n=n, tag=tag, bv=bv, nodes_type=nodes_type, use_std_def=use_std_def)


class CubicSpline(Spline):
    '''
    This class provides a spline object with piecewise cubic polynomials.
    '''
    
    _poly_order = 3
    
    def __init__(self, a=0.0, b=1.0, n=10, bv=dict(), nodes_type='equidistant', tag='', use_std_def=False):
        Spline.__init__(self, a=a, b=b, n=n, tag=tag, bv=bv, nodes_type=nodes_type, use_std_def=use_std_def)
            

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
    
    nodes_type : str {equidistant/chebychev}
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
        
        # determine boundary values of the derivative
        # (and its derivatives)
        bv = dict()
        for k, v in S._boundary_values.items():
            if k == 0:
                pass
            else:
                bv[k-1] = v
        
        # create new spline object
        if po == 3:
            dS = CubicSpline(a=a, b=b, n=n, tag=new_tag, bv=bv)
        elif po == 2:
            dS = QuadraticSpline(a=a, b=b, n=n, tag=new_tag, bv=bv)
        elif po == 1:
            dS = LinearSpline(a=a, b=b, n=n, tag=new_tag, bv=bv)
        elif po == 0:
            dS = ConstantSpline(a=a, b=b, n=n, tag=new_tag, bv=bv)
        
        dS._deriv_order = do
        dS._steady_flag = True
        
        # determine the coefficients of the new derivative
        coeffs = S._coeffs.copy()[:,:(po + 1)]
        
        # get the matrices for provisionally evaluation of the spline
        try:
            dep_array = S._dep_array.copy()[:,:(po + 1)]
            dep_array_abs = S._dep_array_abs.copy()[:,:(po + 1)]
        
            # now consider factors that result from differentiation
            if S.is_cubic:
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
                
                    dep_array[:,0] *= 6
                    dep_array_abs[:,0] *= 6
            elif S.is_quadratic:
                if d == 1:
                    coeffs[:,0] *= 2
                
                    dep_array[:,0] *= 2
                    dep_array_abs[:,0] *= 2
        except AttributeError:
            dep_array = None
            dep_array_abs = None
        
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
    
    coeffs = S._coeffs
    h = S._h
    po = S._poly_order

    # nu represents degree of boundary conditions
    nu = -1
    for k, v in S._boundary_values.items():
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
    # becaus it can't be made 'steady' but the compliance with
    # the boundary values can be enforced
    if S.is_constant:
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
            dep_array_abs[0,0] = S._boundary_values[0][0]
            dep_array_abs[-1,0] = S._boundary_values[0][1]
        
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
    
    if S.is_cubic:
        if nu == -1:
            a = np.hstack([coeffs[:,0], coeffs[0,1:]])
        elif nu == 0:
            a = np.hstack([coeffs[:,0], coeffs[0,2]])
        elif nu == 1:
            a = coeffs[:-1,0]
        elif nu == 2:
            a = coeffs[:-3,0]
    elif S.is_quadratic:
        if nu == -1:
            a = np.hstack([coeffs[:,0], coeffs[0,1:]])
        elif nu == 0:
            a = coeffs[:,0]
        elif nu == 1:
            a = coeffs[:-2,0]
    elif S.is_linear:
        if nu == -1:
            a = np.hstack([coeffs[:,0], coeffs[0,1]])
        elif nu == 0:
            a = coeffs[:-1,0]
    elif S.is_constant:
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
    
    if S.is_constant:
        raise NotImplementedError()
    
    if S._use_std_def:
        for k in xrange(n-1):
            if S.is_cubic:
                block = np.array([[  h[k]**3, h[k]**2,  h[k], 1.0, 0.0, 0.0, 0.0, -1.0],
                                  [3*h[k]**2,  2*h[k],  1.0,  0.0, 0.0, 0.0, -1.0, 0.0],
                                  [  6*h[k],    2.0,    0.0,  0.0, 0.0, -2.0, 0.0, 0.0]])
            elif S.is_quadratic:
                block = np.array([[h[k]**2, h[k], 1.0, 0.0, 0.0, -1.0],
                                  [ 2*h[k],  1.0, 0.0, 0.0, -1.0, 0.0]])
            elif S.is_linear:
                block = np.array([[h[k], 1.0, 0.0, -1.0]])
        
            M[po*k:po*(k+1),(po+1)*k:(po+1)*(k+2)] = block
    
        # add equations for boundary conditions
        if S._boundary_values.has_key(0) and not any(item is None for item in S._boundary_values[0]):
            M[po*(n-1),0:(po+1)] =   np.array([   0.0,       0.0,     0.0,   1.0])
            M[po*(n-1)+1,-(po+1):] = np.array([h[n-1]**3, h[n-1]**2, h[n-1], 1.0])
            r[po*(n-1)] = S._boundary_values[0][0]
            r[po*(n-1)+1] = S._boundary_values[0][1]
        if S._boundary_values.has_key(1) and not any(item is None for item in S._boundary_values[1]):
            M[po*(n-1)+2,0:(po+1)] = np.array([    0.0,       0.0,    1.0, 0.0])
            M[po*(n-1)+3,-(po+1):] = np.array([3*h[n-1]**2, 2*h[n-1], 1.0, 0.0])
            r[po*(n-1)+2] = S._boundary_values[1][0]
            r[po*(n-1)+3] = S._boundary_values[1][1]
        if S._boundary_values.has_key(2) and not any(item is None for item in S._boundary_values[2]):
            M[po*(n-1)+4,0:(po+1)] = np.array([  0.0,    2.0, 0.0, 0.0])
            M[po*(n-1)+5,-(po+1):] = np.array([6*h[n-1], 2.0, 0.0, 0.0])
            r[po*(n-1)+4] = S._boundary_values[2][0]
            r[po*(n-1)+5] = S._boundary_values[2][1]
    else:
        for k in xrange(n-1):
            if S.is_cubic:
                block = np.array([[0.0, 0.0, 0.0, 1.0,   h[k]**3, -h[k]**2, h[k], -1.0],
                                  [0.0, 0.0, 1.0, 0.0, -3*h[k]**2,  2*h[k], -1.0,  0.0],
                                  [0.0, 2.0, 0.0, 0.0,   6*h[k],    -2.0,    0.0,  0.0]])
            elif S.is_quadratic:
                block = np.array([[0.0, 0.0, 1.0, -h[k]**2, h[k], -1.0],
                                  [0.0, 1.0, 0.0,  2*h[k],  -1.0, 0.0]])
            elif S.is_linear:
                block = np.array([[0.0, 1.0, h[k], -1.0]])
        
            M[po*k:po*(k+1),(po+1)*k:(po+1)*(k+2)] = block
        
        # add equations for boundary conditions
        if S._boundary_values.has_key(0) and not any(item is None for item in S._boundary_values[0]):
            M[po*(n-1),0:(po+1)] =   np.array([-h[n-1]**3, h[n-1]**2, -h[n-1], 1.0])
            M[po*(n-1)+1,-(po+1):] = np.array([    0.0,        0.0,     0.0,   1.0])
            r[po*(n-1)] = S._boundary_values[0][0]
            r[po*(n-1)+1] = S._boundary_values[0][1]
        if S._boundary_values.has_key(1) and not any(item is None for item in S._boundary_values[1]):
            M[po*(n-1)+2,0:(po+1)] = np.array([3*h[n-1]**2, -2*h[n-1], 1.0, 0.0])
            M[po*(n-1)+3,-(po+1):] = np.array([    0.0,        0.0,    1.0, 0.0])
            r[po*(n-1)+2] = S._boundary_values[1][0]
            r[po*(n-1)+3] = S._boundary_values[1][1]
        if S._boundary_values.has_key(2) and not any(item is None for item in S._boundary_values[2]):
            M[po*(n-1)+4,0:(po+1)] = np.array([-6*h[n-1], 2.0, 0.0, 0.0])
            M[po*(n-1)+5,-(po+1):] = np.array([   0.0,    2.0, 0.0, 0.0])
            r[po*(n-1)+4] = S._boundary_values[2][0]
            r[po*(n-1)+5] = S._boundary_values[2][1]
    
    return M, r


def interpolate(S=None, fnc=None, points=None, **kwargs):
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
    '''
    '''
    n_nodes : int
        The number of nodes that the interpolating spline should have
        (if the interpolant `S` is not given).
    
    nodes_type : str
        The type of the spline nodes (if the interpolant `S` is not given).
    
    spline_order : int
        The polynomial order of the spline parts (if the interpolant `S` is not given).
    
    '''
    
    params = {'n_nodes' : 100,
              'nodes_type' : 'equidistant',
              'spline_order' : 3,
              'use_std_def' : True}
    
    # check for given keyword arguments
    for k, v in kwargs.items():
        try:
            params[k] = v
        except KeyError:
            pass
    
    if isinstance(S, Spline):
        params['n_nodes'] = S.n + 1
        params['nodes_type'] = S._nodes_type
        params['spline_order'] = S._poly_order
        params['use_std_def'] = S._use_std_def
    
    # first check passed arguments
    try:
        points = np.array(points)
    except Exception as err:
        logging.error('Input argument `points` should be array_like!')
        raise err
    
    if points.ndim == 1:
        # `points` is assumed to contain either interval borders or interpolation nodes
        # so `fnc` has to be given and callable
        assert callable(fnc)
        
        if len(points) == 2:
            # `points` is assumed to contain interval borders so the interpolation nodes
            # have to be generated
            a, b = points
            nodes = np.linspace(a, b, params['n_nodes'], endpoint=True)
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
            raise ValueError('Can only interpolate 1-dimensional function, not {}-dimensional.'.format(values.ndim))
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
        spline_was_given = False
        
        if params['spline_order'] == 0:
            S = ConstantSpline(a=nodes[0], b=nodes[-1], n=nodes.size, 
                                nodes_type=params['nodes_type'], use_std_def=params['use_std_def'])
        elif params['spline_order'] == 1:
            S = LinearSpline(a=nodes[0], b=nodes[-1], n=nodes.size - 1, 
                                nodes_type=params['nodes_type'], use_std_def=params['use_std_def'])
        elif params['spline_order'] == 2:
            S = QuadraticSpline(a=nodes[0], b=nodes[-1], n=nodes.size - 1, 
                                nodes_type=params['nodes_type'], use_std_def=params['use_std_def'])
        elif params['spline_order'] == 3:
            S = CubicSpline(a=nodes[0], b=nodes[-1], n=nodes.size - 1, 
                                nodes_type=params['nodes_type'], use_std_def=params['use_std_def'])
    else:
        spline_was_given = True
        
        # check attributes of the given spline function
        if S.is_constant:
            assert S.n == nodes.size
        else:
            assert S.n == nodes.size - 1
        
        assert S.a == nodes[0] and S.b == nodes[-1]
    
    # choose the interpolation method according to
    # whether the spline has been made steady or not
    if S._steady_flag:
        _interpolate_steady_spline(S, fnc=fnc, points=points)
    else:
        if S._use_std_def:
            _interpolate_non_steady_spline(S, nodes=nodes, values=values)
        else:
            logging.warning('Standard spline interpolation only works ' + 
                            'if spline object uses the standard spline ' +
                            'definition!')
            # solve steadiness and smoothness conditions
            S.make_steady()
            
            # use the other `interpolation` method
            _interpolate_steady_spline(S, fnc=fnc, points=points)
    
    if not spline_was_given:
        return S


def _interpolate_steady_spline(S, fnc=None, points=None):
    '''
    This function is used to determine the free coefficients of a
    spline that has already been made steady such that it sort of
    interpolates a given function or points.
    
    To achieve this the spline function should be equal to the given
    function at specific points in the considered interval.
    
    Parameters
    ----------
    
    S : Spline
        The spline function object used as an interpolant.
    
    fnc : callable
        The function that should be interpolated.
    
    points : array_like
        Two dimensional array containing points that should be interpolated.
    
    '''
    
    assert S._steady_flag
    
    # check if the spline should interpolate a function or given points
    if callable(fnc):
        # how many independent coefficients does the spline have
        coeffs_size = S._indep_coeffs.size
        
        # generate points to evaluate the function at
        # (function and spline interpolant should be equal in these)
        #points = np.linspace(S.a, S.b, coeffs_size, endpoint=False)
        nodes = np.linspace(S.a, S.b, coeffs_size, endpoint=True)
        
        # evaluate the function
        fnc_t = np.array([fnc(t) for t in nodes])
        
        dep_vecs = [S.get_dependence_vectors(t) for t in nodes]
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
        if shape[0] >= shape[1]:
            nodes = points[:,0]
            values = points[:,1]
        else:
            nodes = points[0,:]
            values = points[1,:]
        
        assert S.a <= nodes[0] and nodes[-1] <= S.b
        
        # get dependence matrices of the spline's coefficients
        dep_vecs = [S.get_dependence_vectors(t) for t in nodes]
        S_dep_mat = np.array([vec[0] for vec in dep_vecs])
        S_dep_mat_abs = np.array([vec[1] for vec in dep_vecs])
        
        # solve the equation system
        #free_coeffs = np.linalg.solve(S_dep_mat, fnc_t - S_dep_mat_abs)
        free_coeffs = np.linalg.lstsq(S_dep_mat, values - S_dep_mat_abs)[0]
        
    # set solution for the free coefficients
    S.set_coefficients(free_coeffs=free_coeffs)


def _interpolate_non_steady_spline(S, nodes, values):
    '''
    '''
    
    # set up and solve the interpolation equation system
    if S.is_cubic:
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
        D = sparse.dia_matrix((data, offsets), shape=(S.n+1, S.n+1))
        
        # solve the equation system
        sol = sparse.linalg.spsolve(D.tocsr(),r)
        
        # calculate the coefficients
        coeffs = np.zeros((S.n, 4))
        
        for i in xrange(S.n):
            coeffs[i, :] = [-2.0/h[i]**3 * (values[i+1]-values[i]) + 1.0/h[i]**2 * (sol[i]+sol[i+1]),
                         3.0/h[i]**2 * (values[i+1]-values[i]) - 1.0/h[i] * (2*sol[i]+sol[i+1]),
                         sol[i],
                         values[i]]
    
    elif S.is_quadratic:
        # let `t[i]` be the i-th node and let `z[i]` be the value of the i-th spline part at `t[i]`
        # then it is easy to verify that for i = 0,...,n-1
        # 
        #              (z[i+1] - z[i])                2
        # S[i](t) = --------------------- * (t - t[i])  + z[i] * (t - t[i]) + y[i]
        #           (2 * (t[i+1] - t[i]))
        # 
        # where `y[i]` is the value the spline should take at `t[i]`
        
        # given a `z[0]` we can construct the rest, using the condition S[i](t[i+1]) = y[i+1]
        z = np.zeros(S.n + 1)
        
        z[0] = (values[1] - values[0]) / (nodes[1] - nodes[0])
        for i in xrange(S.n):
            z[i+1] = -z[i] + 2.0 * (values[i+1] - values[i]) / (nodes[i+1] - nodes[i])
        
        # calculate resulting coefficients
        coeffs = np.zeros((S.n, 3))
        
        for i in xrange(S.n):
            coeffs[i, :] = [(z[i+1] - z[i]) / (2.0 * (nodes[i+1] - nodes[i])),
                             z[i],
                             values[i]]
    
    elif S.is_linear:
        coeffs = np.zeros((S.n, 2))
        
        for i in xrange(S.n):
            coeffs[i,0] = (values[i+1] - values[i]) / (nodes[i+1] - nodes[i])
        coeffs[:,1] = values[:-1]
    
    elif S.is_constant:
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
    
    S._steady_flag = True
    
    
    
        


if __name__=='__main__':
    CS = CubicSpline()
    
    if 1:
        S = interpolate(fnc=np.sin, points=np.linspace(0,2*np.pi,10,endpoint=True), spline_order=3)
        tt = np.linspace(0,2*np.pi,1000)
        sint = [np.sin(t) for t in tt]
        
        St = [S(t) for t in tt]
        
        
        #idx = [(int(p[-3]),int(p[-1])) for p in str(CS._indep_coeffs)[1:-1].split(' ')]
    IPS()