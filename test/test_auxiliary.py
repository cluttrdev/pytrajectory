# IMPORTS

import pytrajectory
import pytest
import sympy as sp
import numpy as np

def test_cse_lambdify_sym():
    x, y, z = sp.symbols('x, y, z')

    F = sp.Matrix([(x+y) * (y-z),
                   sp.sin(-(x+y)) + sp.cos(-y+z),
                   sp.exp(sp.sin(-(x+y)) + sp.cos(-y+z))])

    f = pytrajectory.auxiliary.cse_lambdify(args=(x,y,z), expr=F, modules='sympy')

    assert f(x,y,z) == F

def test_cse_lambdify_num():
    x, y, z = sp.symbols('x, y, z')

    F = sp.Matrix([(x+y) * (y-z),
                   sp.sin(-(x+y)) + sp.cos(z-y),
                   sp.exp(sp.sin(-y-x) + sp.cos(-y+z))])

    f = pytrajectory.auxiliary.cse_lambdify(args=(x,y,z), expr=F,
                                            modules=[{'ImmutableMatrix' : np.array}, 'numpy'])

    f_num = f(1.0, 2.0, 3.0)

    assert type(f_num)== np.ndarray
    assert np.allclose(f_num, np.array([[-3.0],
                              [-np.sin(3.0) + np.cos(1.0)],
                              [np.exp(-np.sin(3.0) + np.cos(1.0))]]))

def test_cse_lambdify_num_vectorized():
    x, y, z = sp.symbols('x, y, z')

    F = sp.Matrix([(x+y) * (y-z),
                   sp.sin(-(x+y)) + sp.cos(z-y),
                   sp.exp(sp.sin(-y-x) + sp.cos(-y+z))])

    f = pytrajectory.auxiliary.cse_lambdify(args=(x,y,z), expr=F,
                                            modules=[{'ImmutableMatrix' : np.array}, 'numpy'])

    f_num = f(np.r_[[1.0]*10], np.r_[[2.0]*10], np.r_[[3.0]*10])
    f_num_check = np.array([[-3.0],
                            [-np.sin(3.0) + np.cos(1.0)],
                            [np.exp(-np.sin(3.0) + np.cos(1.0))]])
    
    assert type(f_num)== np.ndarray
    assert np.allclose(f_num, np.tile(f_num_check, (1,10))[:,np.newaxis,:])

