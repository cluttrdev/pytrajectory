# oscillation of the inverted double pendulum with partial linearization

# import trajectory class and necessary dependencies
from pytrajectory.trajectory import Trajectory
from sympy import cos, sin
import numpy as np

# define the function that returns the vectorfield
def f(x,u):
	x1, x2, x3, x4, x5, x6 = x  # system variables
	u, = u                      # input variable
    
    # length of the pendulums
	l1 = 0.7
	l2 = 0.5
    
	g = 9.81    # gravitational acceleration
    
	ff = np.array([         x2,
                            u,
                            x4,
                (1/l1)*(g*sin(x3)+u*cos(x3)),
                            x6,
                (1/l2)*(g*sin(x5)+u*cos(x5))
                    ])
    
	return ff

# system state boundary values for a = 0.0 [s] and b = 2.0 [s]
xa = [0.0, 0.0,  np.pi, 0.0,  np.pi, 0.0]
xb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# boundary values for the input
ua = [0.0]
ub = [0.0]

# create trajectory object
T = Trajectory(f, a=0.0, b=2.0, xa=xa, xb=xb, ua=ua, ub=ub)

# alter some method parameters to increase performance
T.setParam('su', 10)
T.setParam('eps', 8e-2)

# run iteration
T.startIteration()


# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
do_animation = False

if do_animation:
    import matplotlib as mpl
    from pytrajectory.utilities import Animation
    
    def draw(xti, image):
        x, phi1, phi2 = xti[0], xti[2], xti[4]
        
        l1 = 0.7
        l2 = 0.5
    
        car_width = 0.05
        car_heigth = 0.02
        pendel_size = 0.015
    
    
        x_car = x
        y_car = 0
    
        x_pendel1 = -l1*sin(phi1)+x_car
        y_pendel1 = l1*cos(phi1)
    
        x_pendel2 = -l2*sin(phi2)+x_car
        y_pendel2 = l2*cos(phi2)
    
        
        # pendulums
        sphere1 = mpl.patches.Circle((x_pendel1,y_pendel1),pendel_size,color='k')
        sphere2 = mpl.patches.Circle((x_pendel2,y_pendel2),pendel_size,color='0.3')
        
        # car
        car = mpl.patches.Rectangle((x_car-0.5*car_width,y_car-car_heigth),car_width,car_heigth,fill=True,facecolor='0.75',linewidth=2.0)
        
        # joint
        joint = mpl.patches.Circle((x_car,0),0.005,color='k')
        
        # rods
        rod1 = mpl.lines.Line2D([x_car,x_pendel1],[y_car,y_pendel1],color='k',zorder=1,linewidth=2.0)
        rod2 = mpl.lines.Line2D([x_car,x_pendel2],[y_car,y_pendel2],color='0.3',zorder=1,linewidth=2.0)
        
        image.patches.append(sphere1)
        image.patches.append(sphere2)
        image.patches.append(car)
        image.patches.append(joint)
        image.lines.append(rod1)
        image.lines.append(rod2)
        
        return image
    
    A = Animation(drawfnc=draw, simdata=T.sim)
    A.set_limits(xlim=(-1.0,0.8), ylim=(-0.8,0.8))
    
    A.animate()
    A.save('ex2_InvertedDoublePendulumUpswing.mp4')
