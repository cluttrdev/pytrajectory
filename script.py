import pickle
import numpy as np
import matplotlib.pyplot as plt
import pytrajectory
from IPython import embed as IPS
from scipy import sparse


with open('old_spline_x3.pcl', 'r') as pfile:
    pdict = pickle.load(pfile)

coeffs = pdict['x3_old_coeffs']

S = pytrajectory.Spline(a=0.0, b=5.0, n=160)

S.set_coefficients(coeffs=coeffs)


S_new = pytrajectory.Spline(a=0.0, b=5.0, n=320, bv={0:[np.pi, 0.0]})
S_new.make_steady()

if 0:
    points = np.linspace(0.0, 5.0, S_new._indep_coeffs.size+1, endpoint=False)[1:]

    S_old_t = np.array([S.f(t) for t in points])

    dep_vecs = [S_new.get_dependence_vectors(t) for t in points]
    S_new_t = np.array([vec[0] for vec in dep_vecs])
    S_new_t_abs = np.array([vec[1] for vec in dep_vecs])

    sol = np.linalg.solve(S_new_t, S_old_t - S_new_t_abs)

    S_new.set_coefficients(free_coeffs=sol)
else:
    values = [S.f(t) for t in S_new.nodes]

    # create vector of step sizes
    h = np.array([S_new.nodes[k+1] - S_new.nodes[k] for k in xrange(S_new.nodes.size-1)])

    # create diagonals for the coefficient matrix of the equation system
    l = np.array([h[k+1] / (h[k] + h[k+1]) for k in xrange(S_new.nodes.size-2)])
    d = 2.0*np.ones(S_new.nodes.size-2)
    u = np.array([h[k] / (h[k] + h[k+1]) for k in xrange(S_new.nodes.size-2)])

    # right hand site of the equation system
    r = np.array([(3.0/h[k])*l[k]*(values[k+1] - values[k]) + (3.0/h[k+1])*u[k]*(values[k+2]-values[k+1])\
                    for k in xrange(S_new.nodes.size-2)])

    # add conditions for unique solution
    # 
    # natural spline
    #l = np.hstack([l, 1.0, 0.0])
    #d = np.hstack([2.0, d, 2.0])
    #u = np.hstack([0.0, 1.0, u])
    #r = np.hstack([(3.0/h[0])*(values[1]-values[0]), r, (3.0/h[-1])*(values[-1]-values[-2])])

    # boundary derivatives
    l = np.hstack([l, 0.0, 0.0])
    d = np.hstack([1.0, d, 1.0])
    u = np.hstack([0.0, 0.0, u])

    m0 = S.df(S.a)
    mn = S.df(S.b)
    r = np.hstack([m0, r, mn])
    
    data = [l,d,u]
    offsets = [-1, 0, 1]

    # create tridiagonal coefficient matrix
    D = sparse.dia_matrix((data, offsets), shape=(S_new.n+1, S_new.n+1))
    
    # solve the equation system
    sol = sparse.linalg.spsolve(D.tocsr(),r)

    # calculate the coefficients
    coeffs = np.zeros((S_new.n, 4))

    for i in xrange(S_new.n):
        coeffs[i, :] = [-2.0/h[i]**3 * (values[i+1]-values[i]) + 1.0/h[i]**2 * (sol[i]+sol[i+1]),
                     3.0/h[i]**2 * (values[i+1]-values[i]) - 1.0/h[i] * (2*sol[i]+sol[i+1]),
                     sol[i],
                     values[i]]
    
    S_new.set_coefficients(coeffs=coeffs)

tt = np.linspace(0.0, 5.0, 10000)

St = np.array([S.f(t) for t in tt])
St_new = np.array([S_new.f(t) for t in tt])

plt.plot(tt,St, tt[:-100], St_new[:-100])
#plt.plot(points, [S.f(p) for p in points], 'o')
plt.figure()
plt.plot(tt, St-St_new)
plt.plot(S.nodes, S.nodes*0, 'o')

plt.show()


IPS()
