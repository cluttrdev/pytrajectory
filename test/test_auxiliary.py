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

    f = pytrajectory.auxiliary.cse_lambdify(args=(x,y,z), expr=F, modules='numpy')

    f_num = np.array(f(1.0, 2.0, 3.0))
    
    assert np.abs( f_num \
                  - np.array([[-3.0],
                              [-np.sin(3.0) + np.cos(1.0)],
                              [np.exp(-np.sin(3.0) + np.cos(1.0))]])).max() \
           <= np.sqrt(np.finfo(f_num.dtype).eps)

    
