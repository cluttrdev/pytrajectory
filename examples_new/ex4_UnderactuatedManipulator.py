# underactuated manipulator

# import trajectory class and necessary dependencies
from pytrajectory.trajectory import Trajectory
import numpy as np
from sympy import cos, sin

# define the function that returns the vectorfield
def f(x,u):
    x1, x2, x3, x4  = x     # state variables
    u1, = u                 # input variable
    
    e = 0.9     # inertia coupling
    
    s = sin(x3)
    c = cos(x3)
    
    ff = np.array([         x2,
                            u1,
                            x4,
                    -e*x2**2*s-(1+e*c)*u1
                    ])
    
    return ff

# system state boundary values for a = 0.0 [s] and b = 1.8 [s]
xa = [  0.0,
        0.0,
        0.4*np.pi,
        0.0]

xb = [  0.2*np.pi,
        0.0,
        0.2*np.pi,
        0.0]

# boundary values for the inputs
ua = [0.0]
ub = [0.0]

# create trajectory object
T = Trajectory(f, a=0.0, b=1.8, xa=xa, xb=xb, ua=ua, ub=ub)

# also alter some method parameters to increase performance
T.setParam('su', 20)
T.setParam('kx', 3)

# run iteration
T.startIteration()


# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
do_animation = False

if do_animation:
    import matplotlib as mpl
    from pytrajectory.utilities import Animation
    
    def draw(xti, image):
        phi1, phi2 = xti[0], xti[2]
        
        L =0.4
        
        x1 = L*cos(phi1)
        y1 = L*sin(phi1)
        
        x2 = x1+L*cos(phi2+phi1)
        y2 = y1+L*sin(phi2+phi1)
        
        # rods
        rod1 = mpl.lines.Line2D([0,x1],[0,y1],color='k',zorder=0,linewidth=2.0)
        rod2 = mpl.lines.Line2D([x1,x2],[y1,y2],color='k',zorder=0,linewidth=2.0)
        
        # pendulums
        sphere1 = mpl.patches.Circle((x1,y1),0.01,color='k')
        sphere2 = mpl.patches.Circle((0,0),0.01,color='k')
        
        image.lines.append(rod1)
        image.lines.append(rod2)
        image.patches.append(sphere1)
        image.patches.append(sphere2)
        
        return image
    
    A = Animation(drawfnc=draw, simdata=T.sim,
                  plotsys=[(0,'phi1'), (2,'phi2')], plotinputs=[(0,'u')])
    A.set_limits(xlim= (-0.1,0.6), ylim=(-0.4,0.65))
    
    A.animate()
    A.save('ex4_UnderactuatedManipulator.gif')
