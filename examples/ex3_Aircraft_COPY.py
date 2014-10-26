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

# NEW: constraints
con = {4:[-0.6, 0.6]}

# create trajectory object
T = Trajectory(f, a=0.0, b=3.0, xa=xa, xb=xb, g=g, constraints=con)

# don't take advantage of the system structure (integrator chains)
# (this will result in a faster solution here)
T.setParam('use_chains', False)

# also alter some other method parameters to increase performance
#T.setParam('sx', 100)
T.setParam('kx', 3)

# run iteration
T.startIteration()

from IPython import embed as IPS
IPS()

# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
do_animation = True

if do_animation:
    import matplotlib as mpl
    from pytrajectory.utilities import Animation
    
    def draw(xti, image):
        x, y, theta = xti[0], xti[2], xti[4]
        
        S = np.array( [   [0,     0.3],
                          [-0.1,  0.1],
                          [-0.7,  0],
                          [-0.1,  -0.05],
                          [ 0,    -0.1],
                          [0.1,   -0.05],
                          [ 0.7,  0],
                          [ 0.1,  0.1]])
    
        xx = S[:,0].copy()
        yy = S[:,1].copy()
    
        S[:,0] = xx*cos(theta)-yy*sin(theta)+x
        S[:,1] = yy*cos(theta)+xx*sin(theta)+y
    
        aircraft = mpl.patches.Polygon(S, closed=True, facecolor='0.75')
        image.patches.append(aircraft)
        
        return image
    
    A = Animation(drawfnc=draw, simdata=T.sim)
    A.set_limits(xlim=(-1,11), ylim=(-1,21))
    
    A.animate()
    A.save('ex3_Aircraft.mp4')
