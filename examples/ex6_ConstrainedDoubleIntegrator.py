'''
This example of the double integrator demonstrates how to pass constraints to PyTrajectory.
'''
# imports
from pytrajectory.trajectory import Trajectory
import numpy as np

# define the vectorfield
def f(x,u):
    x1, x2 = x
    u1, = u
    
    ff = [x2,
          u1]
    
    return ff

# system state boundary values for a = 0.0 [s] and b = 2.0 [s]
xa = [0.0, 0.0]
xb = [1.0, 0.0]

# constraints dictionary
con = {1 : [-0.1, 0.65]}

# create the trajectory object
T = Trajectory(f, a=0.0, b=2.0, xa=xa, xb=xb, constraints=con, use_chains=False)

# start
x, u = T.startIteration()


# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation

do_animation = False

if do_animation:
    import matplotlib as mpl
    from pytrajectory.utilities import Animation
    
    def draw(xt, image):
        x = xt[0]
        
        car_width = 0.05
        car_heigth = 0.02
        
        x_car = x
        y_car = 0
        
        car = mpl.patches.Rectangle((x_car-0.5*car_width, y_car-car_heigth), car_width, car_heigth,
                                    fill=True, facecolor='grey', linewidth=2.0)
        
        image.patches.append(car)
        
        return image
    
    
    A = Animation(drawfnc=draw, simdata=T.sim,
                        plotsys=[(0,'x'), (1,'dx')],
                        plotinputs=[(0,'u1')])
    xmin = np.min(T.sim[1][:,0])
    xmax = np.max(T.sim[1][:,0])
    A.set_limits(xlim=(xmin - 0.1, xmax + 0.1), ylim=(-0.1,0.1))
    A.animate()
    A.save('ex7_DoubleIntegrator.mp4')
