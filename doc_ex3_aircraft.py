# vertical take-off aircraft

# import trajectory class and necessary dependencies
from pytrajectory import Trajectory
from sympy import sin, cos
import numpy as np
from numpy import pi

# define the function that returns the vectorfield
def f(x,u):
    x1, x2, x3, x4, x5, x6 = x  # system state variables
    u1, u2 = u                  # input variables
    
    # coordinates for the points in which the engines engage [m]
    l = 1.0
    h = 0.1

    g = 9.81    # graviational acceleration [m/s^2]
    M = 50.0    # mass of the aircraft [kg]
    J = 25.0    # moment of inertia about M [kg*m^2]

    alpha = 5/360.0*2*pi    # deflection of the engines

    sa = sin(alpha)
    ca = cos(alpha)

    s = sin(x5)
    c = cos(x5)
    
    ff = np.array([             x2,
                    -s/M*(u1+u2) + c/M*(u1-u2)*sa,
                                x4,
                    -g+c/M*(u1+u2) +s/M*(u1-u2)*sa ,
                                x6,
                    1/J*(u1-u2)*(l*ca+h*sa)])
    
    return ff 

# system state boundary values for a = 0.0 [s] and b = 3.0 [s]
xa = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
xb = [10.0, 0.0, 5.0, 0.0, 0.0, 0.0]

# boundary values for the inputs
g = [0.5*9.81*50.0/(cos(5/360.0*2*pi)),
     0.5*9.81*50.0/(cos(5/360.0*2*pi))]

# create trajectory object
T = Trajectory(f, a=0.0, b=3.0, xa=xa, xb=xb, g=g)

# don't take advantage of the system structure (integrator chains)
# (this will result in a faster solution here)
T.setParam('use_chains', False)

# also alter some other method parameters to increase performance
T.setParam('kx', 5)

# run iteration
T.startIteration()

# show results
T.plot()
