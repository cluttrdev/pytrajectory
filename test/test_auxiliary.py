# IMPORTS

import pytrajectory
import pytest
import sympy as sp
import numpy as np


class TestCseLambdify(object):
    
    def test_list(self):
        x, y = sp.symbols('x, y')
        ones = np.ones(10)
    
        l = [0.5*(x + y), sp.asin(sp.sin(0.5*(x+y))), sp.sin(x+y)**2 + sp.cos(x+y)**2]

        f = pytrajectory.auxiliary.cse_lambdify(args=(x,y), expr=l, modules='numpy')

        assert f(1., 1.) == [1., 1., 1.]
        for i in f(ones, ones):
            assert np.allclose(i, ones)

    def test_matrix_to_matrix(self):
        x, y = sp.symbols('x, y')
        ones = np.ones(10)
    
        M = sp.Matrix([0.5*(x + y), sp.asin(sp.sin(0.5*(x+y))), sp.sin(x+y)**2 + sp.cos(x+y)**2])

        f = pytrajectory.auxiliary.cse_lambdify(args=(x,y), expr=M,
                                                modules='numpy')

        assert type(f(1., 1.)) == np.matrix
        assert np.allclose(f(1. ,1.), np.ones((3,1)))

    def test_matrix_to_array(self):
        x, y = sp.symbols('x, y')
        ones = np.ones(10)
    
        M = sp.Matrix([0.5*(x + y), sp.asin(sp.sin(0.5*(x+y))), sp.sin(x+y)**2 + sp.cos(x+y)**2])

        f = pytrajectory.auxiliary.cse_lambdify(args=(x,y), expr=M,
                                                modules=[{'ImmutableMatrix' : np.array}, 'numpy'])

        F = f(1., 1.)
        
        assert type(F == np.ndarray)
        assert not isinstance(F, np.matrix)
        assert F.shape == (3,1)
        assert np.allclose(F, np.ones((3,1)))

    def test_array_input(self):
        pass
        



