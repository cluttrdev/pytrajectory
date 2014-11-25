'''
Constrained double pendulum
'''

# import all we need for solving the problem
from pytrajectory import Trajectory
import numpy as np
import sympy as sp
from sympy import cos, sin
from numpy import pi

# first, we define the function that returns the vectorfield
def f(x,u):
    x, dx, phi1, dphi1, phi2, dphi2 = x     # system variables
    F, = u                                  # input variable
    
    l1 = 0.5    # length of the pendulum 1
    l2 = 0.5    # length of the pendulum 2
    m1 = 0.1    # mass of the pendulum 1
    m2 = 0.1    # mass of the pendulum 2
    
    I1 = 4.0/3.0 * m1 * l1**2
    I2 = 4.0/3.0 * m2 * l2**2
    
    m = 1.0     # mass of the car
    g = 9.81    # gravitational acceleration
    
    # mass matrix
    M= np.array([[      m+m1+m2,          (m1+2*m2)*l1*cos(phi1),   m2*l2*cos(phi2)],
                 [(m1+2*m2)*l1*cos(phi1),   I1+(m1+4*m2)*l1**2,   2*m2*l1*l2*cos(phi2-phi1)],
                 [  m2*l2*cos(phi2),     2*m2*l1*l2*cos(phi2-phi1),     I2+m2*l2**2]])
    
    # right hand side
    B= np.array([[ F + (m1+2*m2)*l1*sin(phi1)*dphi1**2 + m2*l2*sin(phi2)*dphi2**2 ],
                 [ (m1+2*m2)*g*l1*sin(phi1) + 2*m2*l1*l2*sin(phi2-phi1)*dphi2**2 ],
                 [ m2*g*l2*sin(phi2) + 2*m2*l1*l2*sin(phi1-phi2)*dphi1**2 ]])
    
    if isinstance(x, sp.Symbol):
        ddx, ddphi1, ddphi2 = sp.Matrix(M).solve(sp.Matrix(B))
    else:
        ddx, ddphi1, ddphi2 = np.linalg.solve(M,B)
    
    # this is the vectorfield
    ff = [ dx,
           ddx,
           dphi1,
           ddphi1,
           dphi2,
           ddphi2]
    
    return ff

# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0, pi, 0.0, pi, 0.0]

b = 4.0
xb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

ua = [0.0]
ub = [0.0]

# here we specify the constraints for the velocity of the car
con = {1 : [-5.0, 5.0]}

# now we create our Trajectory object and alter some method parameters via the keyword arguments
T = Trajectory(f, a, b, xa, xb, ua, ub, constraints=con, su=10, kx=5, use_chains=False)

# time to run the iteration
x, u = T.startIteration()


# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
do_animation = False

if do_animation:
    import matplotlib as mpl
    from pytrajectory.utilities import Animation
    
    def draw(xt, image):
        x = xt[0]
        phi1 = xt[2]
        phi2 = xt[4]
    
        car_width = 0.05
        car_heigth = 0.02
    
        rod_length = 0.5
        pendulum_size = 0.015
    
        x_car = x
        y_car = 0
    
        x_pendulum1 = x_car + rod_length * sin(phi1)
        y_pendulum1 = rod_length * cos(phi1)
    
        x_pendulum2 = x_pendulum1 + rod_length * sin(phi2)
        y_pendulum2 = y_pendulum1 + rod_length * cos(phi2)
    
        # create image
        pendulum1 = mpl.patches.Circle(xy=(x_pendulum1, y_pendulum1), radius=pendulum_size, color='black')
        pendulum2 = mpl.patches.Circle(xy=(x_pendulum2, y_pendulum2), radius=pendulum_size, color='black')
        
        car = mpl.patches.Rectangle((x_car-0.5*car_width, y_car-car_heigth), car_width, car_heigth,
                                    fill=True, facecolor='grey', linewidth=2.0)
        joint = mpl.patches.Circle((x_car,0), 0.005, color='black')
        
        rod1 = mpl.lines.Line2D([x_car,x_pendulum1], [y_car,y_pendulum1],
                                color='black', zorder=1, linewidth=2.0)
        rod2 = mpl.lines.Line2D([x_pendulum1,x_pendulum2], [y_pendulum1,y_pendulum2],
                                color='black', zorder=1, linewidth=2.0)
    
        # add the patches and lines to the image
        image.patches.append(pendulum1)
        image.patches.append(pendulum2)
        image.patches.append(car)
        image.patches.append(joint)
        image.lines.append(rod1)
        image.lines.append(rod2)
    
        # and return the image
        return image

    # create Animation object
    A = Animation(drawfnc=draw, simdata=T.sim)
    xmin = np.min(T.sim[1][:,0])
    xmax = np.max(T.sim[1][:,0])
    A.set_limits(xlim=(xmin - 0.5, xmax + 0.5), ylim=(-1.2,1.2))
    
    A.animate()
    A.save('ex8_ConstrainedDoublePendulum.mp4')
