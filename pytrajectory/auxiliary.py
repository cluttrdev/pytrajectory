# IMPORTS
import numpy as np
import sympy as sp
import logging
import time

from IPython import embed as IPS


class Timer(object):
    '''
    Provides a context manager that takes the time of a code block.
    
    Parameters
    ----------
    
    label : str
        The 'name' of the code block which is timed
    
    verb : int
        Level of verbosity
    '''
    def __init__(self, label="~"):
        self.label = label

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.delta = time.time() - self.start
        logging.debug("--> [%s elapsed %f s]"%(self.label, self.delta))


class BetweenDict(dict):
    def __init__(self, d = {}):
        for k, v in d.iteritems():
            self[(k[0], k[1])] = v

    def __getitem__(self, key):
        for k, v in self.iteritems():
            if k[0] <= key < k[1]:
                return v
            
        raise KeyError("Key '%s' is not between any values in the BetweenDict" % key)

    def __setitem__(self, key, value):
        try:
            if len(key) == 2:
                if key[0] < key[1]:
                    dict.__setitem__(self, (key[0], key[1]), value)
                else:
                    raise RuntimeError('First element of a BetweenDict key '
                                       'must be strictly less than the '
                                       'second element')
            else:
                raise ValueError('Key of a BetweenDict must be an iterable '
                                 'with length two')
        except TypeError:
            raise TypeError('Key of a BetweenDict must be an iterable '
                             'with length two')

    def __contains__(self, key):
        try:
            return bool(self[key]) or True
        except KeyError:
            return False
    
    def __str__(self):
        s = '{{ {} }}'
        rep = ''
        for k, v in sorted(self.iteritems(), key=lambda item: item[0]):
            rep += '{} : {}, '.format(k, v)
        
        return s.format(rep[:-1])
    
    def __repr__(self):
        rep = ''
        for k, v in sorted(self.iteritems(), key=lambda item: item[0]):
            rep += '{} : {},\n'.format(k, v)
        
        print '{' + rep + '}'
        return


class IntegChain(object):
    '''
    This class provides a representation of an integrator chain.
    
    For the elements :math:`(x_i)_{i=1,...,n}` of the chain the relation
    :math:`\dot{x}_i = x_{i+1}` applies.
    
    Parameters
    ----------
    
    lst : list
        Ordered list of sympy.symbols for elements of the integrator chain.
    
    Attributes
    ----------
    
    elements : tuple
        Ordered list of all elements that are part of the integrator chain
    
    upper : sympy.Symbol
        Upper end of the integrator chain
    
    lower : sympy.Symbol
        Lower end of the integrator chain
    '''
    
    def __init__(self, lst):
        self._elements = tuple(lst)
        self.upper = self._elements[0]
        self.lower = self._elements[-1]
    
    def __len__(self):
        return len(self._elements)
    
    def __getitem__(self, key):
        return self._elements[key]
    
    def __contains__(self, item):
        return (item in self._elements)
    
    def __str__(self):
        s = ''
        for elem in self._elements:#[::-1]:
            s += ' -> ' + elem.name
        return s[4:]
    
    def elements(self):
        '''
        Return an ordered list of the integrator chain's elements.
        '''
        return self._elements


def find_integrator_chains(fi, x_sym, u_sym):
    '''
    Searches for integrator chains in given vector field matrix `fi`,
    i.e. equations of the form :math:`\dot{x}_i = x_j`.
    
    Parameters
    ----------
    
    fi : array_like
        Matrix representation for the vectorfield of the control system.
    
    x_sym : list
        Symbols for the state variables.
    
    u_sym : list
        Symbols for the input variables.
    
    Returns
    -------
    
    list
        Found integrator chains.
    
    list
        Indices of the equations that have to be solved using collocation.
    '''
    
    n = len(x_sym)
    assert n == len(fi)
    
    chaindict = {}
    for i in xrange(len(fi)):
        # substitution because of sympy difference betw. 1.0 and 1
        if isinstance(fi[i], sp.Basic):
            fi[i] = fi[i].subs(1.0, 1)

        for xx in x_sym:
            if fi[i] == xx:
                chaindict[xx] = x_sym[i]

        for uu in u_sym:
            if fi[i] == uu:
                chaindict[uu] = x_sym[i]

    # chaindict looks like this:  {u_1 : x_2, x_4 : x_3, x_2 : x_1}
    # where x_4 = d/dt x_3 and so on

    # find upper ends of integrator chains
    uppers = []
    for vv in chaindict.values():
        if (not chaindict.has_key(vv)):
            uppers.append(vv)

    # create ordered lists that temporarily represent the integrator chains
    tmpchains = []

    # therefore we flip the dictionary to work our way through its keys
    # (former values)
    dictchain = {v:k for k,v in chaindict.items()}

    for var in uppers:
        tmpchain = []
        vv = var
        tmpchain.append(vv)

        while dictchain.has_key(vv):
            vv = dictchain[vv]
            tmpchain.append(vv)

        tmpchains.append(tmpchain)

    # create an integrator chain object for every temporary chain
    chains = []
    for lst in tmpchains:
        ic = IntegChain(lst)
        chains.append(ic)
        logging.debug("--> found: " + str(ic))
    
    # now we determine the equations that have to be solved by collocation
    # (--> lower ends of integrator chains)
    eqind = []

    if chains:
        # iterate over all integrator chains
        for ic in chains:
            # if lower end is a system variable
            # then its equation has to be solved
            if ic.lower.name.startswith('x'):
                idx = x_sym.index(ic.lower)
                eqind.append(idx)
        eqind.sort()
        
        # if every integrator chain ended with input variable
        if not eqind:
            eqind = range(n)
    else:
        # if integrator chains should not be used
        # then every equation has to be solved by collocation
        eqind = range(n)
    
    
    return chains, eqind


def sym2num_vectorfield(f_sym, x_sym, u_sym, vectorized=False):
    '''
    This function takes a callable vector field of a control system that is to be evaluated with symbols
    for the state and input variables and returns a corresponding function that can be evaluated with
    numeric values for these variables.
    
    Parameters
    ----------
    
    f_sym : callable or array_like
        The callable ("symbolic") vector field of the control system.
    
    x_sym : iterable
        The symbols for the state variables of the control system.
    
    u_sym : iterable
        The symbols for the input variables of the control system.
    
    vectorized : bool
        Whether or not to return a vectorized function.
    
    Returns
    -------
    
    callable
        The callable ("numeric") vector field of the control system.
    '''
    
    # get a sympy.Matrix representation of the vectorfield
    if callable(f_sym):
        F = sp.Matrix(f_sym(x_sym, u_sym))
        if F.T == F.vec():
            F = F.tolist()[0]
        else:
            F = F.T.tolist()[0]
    else:
        try:
            F = sp.Matrix(f_sym)
        except:
            print "ERROR: array_like f_sym to sp.Matrix failed!"
            IPS()
    
    if not vectorized:
        # Use lambdify to replace sympy functions in the vectorfield with
        # numpy equivalents
        _f_num = sp.lambdify(x_sym + u_sym, F, modules='numpy')
        
        # Create a wrapper as the actual function due to the behaviour
        # of lambdify()
        def f_num(x, u):
            xu = np.hstack((x, u))
            return np.array(_f_num(*xu))
    else:
        f_str = repr(np.array(F).ravel()).replace(', dtype=object', '')
        _f_num = sp.lambdify(x_sym + u_sym, f_str, modules='numpy')
        
        # Create a wrapper as the actual function due to the behaviour
        # of lambdify()
        def f_num(x, u):
            xu = np.vstack((x, u))
            return np.array(_f_num(*xu))
    
    return f_num


def saturation_functions(y_fnc, dy_fnc, y0, y1):
    '''
    Creates callable saturation function and its first derivative to project 
    the solution found for an unconstrained state variable back on the original
    constrained one.
    
    For more information, please have a look at :ref:`handling_constraints`.
    
    Parameters
    ----------
    
    y_fnc : callable
        The calculated solution function for an unconstrained variable.
    
    dy_fnc : callable
        The first derivative of the unconstrained solution function.
    
    y0 : float
        Lower saturation limit.
    
    y1 : float
        Upper saturation limit.
    
    Returns
    -------
    
    callable
        A callable of a saturation function applied to a calculated solution
        for an unconstrained state variable.
    
    callable
        A callable for the first derivative of a saturation function applied 
        to a calculated solution for an unconstrained state variable.
    '''
    
    # Calculate the parameter m such that the slope of the saturation function
    # at t = 0 becomes 1
    m = 4.0/(y1-y0)
    
    # this is the saturation function
    def psi_y(t):
        y = y_fnc(t)
        return y1 - (y1-y0)/(1.0+np.exp(m*y))
    
    # and this its first derivative
    def dpsi_dy(t):
        y = y_fnc(t)
        dy = dy_fnc(t)
        return dy * (4.0*np.exp(m*y))/(1.0+np.exp(m*y))**2
    
    return psi_y, dpsi_dy


def consistency_error(I, x_fnc, u_fnc, dx_fnc, ff_fnc, npts=500, return_error_array=False):
    '''
    Calculates an error that shows how "well" the spline functions comply with the system
    dynamic given by the vector field.
    
    Parameters
    ----------
    
    I : tuple
        The considered time interval.
    
    x_fnc : callable
        A function for the state variables.
    
    u_fnc : callable
        A function for the input variables.
    
    dx_fnc : callable
        A function for the first derivatives of the state variables.
    
    ff_fnc : callable
        A function for the vectorfield of the control system.
    
    npts : int
        Number of point to determine the error at.
    
    return_error_array : bool
        Whether or not to return the calculated errors (mainly for plotting).
    
    Returns
    -------
    
    float
        The maximum error between the systems dynamic and its approximation.
    
    numpy.ndarray
        An array with all errors calculated on the interval.
    '''
    
    # get some test points to calculate the error at
    tt = np.linspace(I[0], I[1], npts, endpoint=True)
    
    error = []
    for t in tt:
        x = x_fnc(t)
        u = u_fnc(t)
        
        ff = ff_fnc(x, u)
        dx = dx_fnc(t)
        
        error.append(ff - dx)
    error = np.array(error)
    
    max_con_err = error.max()
    
    if return_error_array:
        return max_con_err, error
    else:
        return max_con_err


if __name__ == '__main__':
    from sympy import sin, cos, exp
    
    def f_sym(x,u):
        x1, x2 = x
        u1, = u

        ff = np.array([ sin(x2),
                        exp(-u1)])
        return ff
    
    def Df_sym(x, u):
        x1, x2 = x
        u1, = u
        
        df = np.array([[0.0, cos(x2),   0.0  ],
                       [0.0,   0.0,  -exp(u1)]])
        
        return df
    
    x_sym = list(sp.symbols('x1, x2'))
    u_sym = list(sp.symbols('u1,'))
    
    # PROBLEM: fails if some element of vector field of jacobian is constant...
    IPS()