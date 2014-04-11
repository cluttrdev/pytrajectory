import numpy as np
from numpy.linalg import solve
import sympy as sp
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

import log
#from log import IPS
from IPython import embed as IPS


def fdiff(func):
    '''
    This function is used to get the derivative of of a callable splinefunction.
    
    
    Parameters
    ----------
    
    func : callable - The spline function to derivate.
    '''
    
    # Return derivative of temporary function
    # im_func is the function's id
    # im_self is the object of which func is the method
    if(func.im_func == CubicSpline.tmp_f.im_func):
        return func.im_self.tmp_df
    if(func.im_func == CubicSpline.tmp_df.im_func):
        return func.im_self.tmp_ddf
    if(func.im_func == CubicSpline.tmp_ddf.im_func):
        return func.im_self.tmp_dddf
    
    # Return derivative of callable function
    if(func.im_func == CubicSpline.f.im_func):
        return func.im_self.df
    if(func.im_func == CubicSpline.df.im_func):
        return func.im_self.ddf
    if(func.im_func == CubicSpline.ddf.im_func):
        return func.im_self.dddf

    # raise notImplemented
    print 'not implemented'
    return None


class CubicSpline():
    '''
    This class provides an object that represents a cubic spline ...
    
    Parameters
    ----------
    
    a : float
        Left border of spline interval.
    b : float
        Right border of spline interval.
    n : int
        Number of polynomial parts the spline will be devided into.
    tag : str
        The 'name' of the spline object.
    bc : tuple
        Boundary values for the spline function itself.
    bcd : tuple
        Boundary values for the splines 1st derivative
    bcdd : tuple
        Boundary values for the splines 2nd derivative
    steady : bool
        Whether or not to call :func:`makesteady()` when instanciated.
    
    '''
    
    def __init__(self, a=0.0, b=1.0, n=10, tag='', bc=None, bcd=None, bcdd=None, steady=True):
        # [a,b] ... interval
        # n     ... number of spline parts
        # tag   ... string with name of the spline object

        self.a = a
        self.b = b
        self.n = int(n)
        self.tag = tag
        self.bc = bc
        self.bcd = bcd
        self.bcdd = bcdd

        # size of polynomial parts
        self.h = (float(b)-float(a))/float(n)

        # create array of symbolic coefficients
        self.coeffs = sp.symarray('c'+tag, (self.n, 4))

        #   key: spline part    value: coefficients of the polynom
        self.S = dict()

        self.tmp_S = np.ones_like(self.coeffs)
        self.tmp_S_abs = np.zeros((self.n,4))

        for i in xrange(self.n):
            # create polynoms:  p_i(x)= c_i_0*x^3 + c_i_1*x^2 + c_i_2*x + c_i_3
            self.S[i] = np.poly1d(self.coeffs[i])

        self.steady_flag = False
        self.tmp_flag = True

        if (steady):
            with log.Timer("makesteady()"):
                self.makesteady()
    
    
    def tmp_evalf(self, x, d):
        '''
        This function returns a matrix and vector to evaluate the spline or a derivative at x
        by multiplying the matrix with numerical values of the independent variables
        and adding the vector.
        
        
        Parameters
        ----------
        
        x : real
            The point to evaluate the spline at
        d : int
            The derivation order
        
        
        Returns
        -------
        
        tuple
            Matrix and vector that represent how the splines coefficients depend on the free parameters.
        '''
        
        # Get the spline part where x is in
        i = int(np.floor(x*self.n/(self.b)))
        if (i == self.n): i-= 1
        
        x -= (i+1)*self.h
        
        # Calculate vector to for multiplication with coefficient matrix w.r.t. the derivation order
        if (d == 0):
            p = np.array([x*x*x,x*x,x,1.0])
        elif (d == 1):
            p = np.array([3.0*x*x,2.0*x,1.0,0.0])
        elif (d == 2):
            p = np.array([6.0*x,2.0,0.0,0.0])
        elif (d == 3):
            p = np.array([6.0,0.0,0.0,0.0])
        
        M0 = np.array([m for m in self.tmp_S[i]],dtype=float)
        m0 = self.tmp_S_abs[i]
        
        return np.dot(p,M0), np.dot(p,m0)
    
    
    def evalf(self, x, d):
        '''
        Returns the value of the splines :attr:`d`-th derivative at :attr:`x`.
        
        
        Parameters
        ----------
        
        x : float
            The point to evaluate the spline at
        d : int
            The derivation order
        '''
        
        # get polynomial part where x is in
        i = int(np.floor(x*self.n/(self.b)))
        if (i == self.n): i-= 1
        p = self.S[i]

        return p.deriv(d)(x-(i+1)*self.h)
    
    def f(self, x):
        '''This is just a wrapper for :meth:`evalf` to evaluate the spline itself.'''
        if self.tmp_flag:
            return self.tmp_evalf(x,0)
        else:
            return self.evalf(x,0)

    def df(self, x):
        '''This is just a wrapper for :meth:`evalf` to evaluate the splines 1st derivative.'''
        if self.tmp_flag:
            return self.tmp_evalf(x,1)
        else:
            return self.evalf(x,1)

    def ddf(self, x):
        '''This is just a wrapper for :meth:`evalf` to evaluate the splines 2nd derivative.'''
        if self.tmp_flag:
            return self.tmp_evalf(x,2)
        else:
            return self.evalf(x,2)

    def dddf(self, x):
        '''This is just a wrapper for :meth:`evalf` to evaluate the splines 3rd derivative.'''
        if self.tmp_flag:
            return self.tmp_evalf(x,3)
        else:
            return self.evalf(x,3)
    
    
    def makesteady(self):
        '''
        This method sets up and solves equations that satisfy boundary conditions and
        ensure steadiness and smoothness conditions of the spline in every joining point.
        '''
        
        log.info("makesteady: "+self.tag)
        
        # This should be untouched yet
        assert self.steady_flag == False

        coeffs = self.coeffs
        h = self.h

        # mu represents degree of boundary conditions
        mu = -1
        if (self.bc != None):
            mu += 1
        if (self.bcd != None):
            mu += 1
        if (self.bcdd != None):
            mu += 1

        # ---> docu p. 14
        v= 0
        if (mu == -1):
            a = coeffs[:,v]
            a = np.hstack((a,coeffs[0,list({0,1,2,3}-{v})]))
        if (mu == 0):
            a = coeffs[:,v]
            a = np.hstack((a,coeffs[0,2]))
        if (mu == 1):
            a = coeffs[:-1,v]
        if (mu == 2):
            a = coeffs[:-3,v]

        # b is what is not in a
        coeffs_set = set(coeffs.flatten())
        a_set = set(a)
        b_set = coeffs_set - a_set
        
        # transfer b_set to ordered list
        b = sorted(list(b_set), key = lambda c: c.name)
        #b = np.array(sorted(list(b_set), key = lambda c: c.name))
        
        # Build Matrix for equation system of smoothness conditions --> p. 13

        # get matrix dimensions --> (3.21) & (3.22)
        N2 = 4*self.n
        N1 = 3*(self.n-1) + 2*(mu+1)

        M = np.zeros((N1,N2))
        r = np.zeros(N1)

        # build block band matrix for smoothness in every joining point
        #   derivatives from order 0 to 2
        #   --> (3.19), caution because of sign error in documentation
        repmat = np.array([[0.0, 0.0, 0.0, 1.0,   h**3, -h**2,  h,  -1.0],
                           [0.0, 0.0, 1.0, 0.0, -3*h**2, 2*h, -1.0, 0.0],
                           [0.0, 2.0, 0.0, 0.0,    6*h,  -2.0,  0.0, 0.0]])

        for i in xrange(self.n-1):
            M[3*i:3*(i+1),4*i:4*(i+2)] = repmat

        # add equations for boundary conditions
        if (self.bc != None):
            M[3*(self.n-1),0:4] = np.array([-h**3, h**2, -h, 1.0])
            M[3*(self.n-1)+1,-4:] = np.array([0.0, 0.0, 0.0, 1.0])
            r[3*(self.n-1)] = self.bc[0]
            r[3*(self.n-1)+1] = self.bc[1]
        if (self.bcd != None):
            M[3*(self.n-1)+2,0:4] = np.array([3*h**2, -2*h, 1.0, 0.0])
            M[3*(self.n-1)+3,-4:] = np.array([0.0, 0.0, 1.0, 0.0])
            r[3*(self.n-1)+2] = self.bcd[0]
            r[3*(self.n-1)+3] = self.bcd[1]
        if (self.bcdd != None):
            M[3*(self.n-1)+4,0:4] = np.array([-6*h, 2.0, 0.0, 0.0])
            M[3*(self.n-1)+5,-4:] = np.array([0.0, 2.0, 0.0, 0.0])
            r[3*(self.n-1)+4] = self.bcdd[0]
            r[3*(self.n-1)+5] = self.bcdd[1]

        # get A and B matrix --> docu p. 13
        #
        #       M*c = r
        # A*a + B*b = r
        #         b = B^(-1)*(r-A*a)
        #
        # we need B^(-1)*r [absolute part -> tmp1] and B^(-1)*A [coefficients of a -> tmp2]

        a_mat = np.zeros((N2,N2-N1))
        b_mat = np.zeros((N2,N1))
        
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
        A = M.dot(a_mat)
        B = M.dot(b_mat)

        # do the inversion
        tmp1 = np.array(solve(B,r.T),dtype=np.float)
        tmp2 = np.array(solve(B,-A),dtype=np.float)

        tmp_coeffs = np.zeros_like(self.coeffs, dtype=None)
        tmp_coeffs_abs = np.zeros((self.n,4))

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

        self.tmp_S = tmp_coeffs
        self.tmp_S_abs = tmp_coeffs_abs

        # docu p. 13: a is vector of independent spline coeffs
        self.c_indep = a

        self.steady_flag = True
    
    
    def set_coeffs(self, c_sol):
        '''
        This function is used to set up numerical values for the spline coefficients.
        
        
        Parameters
        ----------
        
        c_sol : numpy.ndarray
            Array with numerical values for the free spline coefficients
        '''
        
        for i in xrange(self.n):
            c_num = [np.dot(c,c_sol)+ca for c,ca in zip(self.tmp_S[i],self.tmp_S_abs[i])]
            self.S[i] = np.poly1d(c_num)
        
        self.tmp_flag = False
