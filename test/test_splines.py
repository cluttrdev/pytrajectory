


import numpy as np
import pytest

from pytrajectory.splines import Spline


class TestSpline(object):
    a = 0.
    b = 1.
    n = 10

    bv= {0 : [1., 1.],
         1 : [-1., -1.]}

    S = Spline(a=a, b=b, n=n, bv=bv)

    def test_dependence_vectors(self):
        # first make steady
        S.make_steady()

        # set indep coeffs to 1
        ones = np.ones(S._indep_coeffs.size)
        S.set_coefficients(free_coeffs=ones)

        points = np.linspace(start=S.a, stop=S.b, num=3*S.n, endpoint=True)

        val_0 = np.array([S.f(p) for p in points])

        val_1 = []
        for p in points:
            mx, mx_abs = S.get_dependence_vectors(p)
            val_1.append(mx.dot(ones) + mx_abs)
        val_1 = np.array(val_1)
        
