'''
This example of the inverted pendulum demonstrates the basic usage of
PyTrajectory as well as its visualisation capabilities.
'''

# import all we need for solving the problem
from pytrajectory import Trajectory
import numpy as np
from sympy import cos, sin
from numpy import pi

# the next imports are necessary for the visualisatoin of the system
import matplotlib as mpl
from pytrajectory.utilities import Animation


# first, we define the function that returns the vectorfield
def f(x,u):
    x1, x2, x3, x4 = x  # system variables
    u1, = u             # input variable
    
    l = 0.5     # length of the pendulum
    g = 9.81    # gravitational acceleration
    
    # this is the vectorfield
    ff = [          x2,
                    u1,
                    x4,
            (1/l)*(g*sin(x3)+u1*cos(x3))]
    
    return ff

# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0, pi, 0.0]

b = 2.0
xb = [0.0, 0.0, 0.0, 0.0]

uab = [0.0, 0.0]

c = {0:[-0.9, 0.2]}

# now we create our Trajectory object and alter some method parameters via the keyword arguments
T = Trajectory(f, a, b, xa, xb, uab, sx=250, kx=5, use_chains=True, constraints=c)

# time to run the iteration
T.startIteration()



# now that we (hopefully) have found a solution,
# we can visualise our systems dynamic

# therefor we define a fnuction that draws a image of the system
# according to the given simulation data

def draw(xt, image):
    # to draw the image we just need the translation `x` of the
    # cart and the deflection angle `phi` of the pendulum.
    x = xt[0]
    phi = xt[2]
    
    # next we set some parameters
    car_width = 0.05
    car_heigth = 0.02
    
    rod_length = 0.5
    pendulum_size = 0.015
    
    # then we determine the current state of the system
    # according to the given simulation data
    x_car = x
    y_car = 0
    
    x_pendulum = -rod_length * sin(phi) + x_car
    y_pendulum = rod_length * cos(phi)
    
    # now we can build the image
    
    # the pendulum will be represented by a black circle with
    # center: (x_pendulum, y_pendulum) and radius `pendulum_size
    pendulum = mpl.patches.Circle(xy=(x_pendulum, y_pendulum), radius=pendulum_size, color='black')
    
    # the cart will be represented by a grey rectangle with
    # lower left: (x_car - 0.5 * car_width, y_car - car_heigth)
    # width: car_width
    # height: car_height
    car = mpl.patches.Rectangle((x_car-0.5*car_width, y_car-car_heigth), car_width, car_heigth,
                                fill=True, facecolor='grey', linewidth=2.0)
    
    # the joint will also be a black circle with
    # center: (x_car, 0)
    # radius: 0.005
    joint = mpl.patches.Circle((x_car,0), 0.005, color='black')
    
    # and the pendulum rod will just by a line connecting the cart and the pendulum
    rod = mpl.lines.Line2D([x_car,x_pendulum], [y_car,y_pendulum],
                            color='black', zorder=1, linewidth=2.0)
    
    # finally we add the patches and line to the image
    image.patches.append(pendulum)
    image.patches.append(car)
    image.patches.append(joint)
    image.lines.append(rod)
    
    # and return the image
    return image


# now we can create an instance of the `Animation` class 
# with our draw function and the simulation results
A = Animation(drawfnc=draw, simdata=T.sim)

# as for now we have to explicitly set the limits of the figure
# (may involves some trial and error)
#A.set_limits(xlim=(-1.2,0.3), ylim=(-0.6,0.6))
A.set_limits(xlim=(-1.2,0.3), ylim=(-0.6,0.6))

# if everything is set, we can start the animation
# (might take some while)
A.animate()

# then we can save the animation as a `mp4` video file or as an animated `gif` file
A.save('simple_example_copy.mp4')
