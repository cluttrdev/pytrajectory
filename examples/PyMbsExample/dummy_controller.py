# This file contains the dummy procedure for the controller that is used
# in the example for the interaction between PyMbs and PyTrajectory.

import numpy as np

# First we define a dummy procedure
def u(t):
    # The system is 3-dimensional so we just return the vector (0,0,0)
    return np.zeros(3)

# We store this dummy procedure in a container so that we can easily
# replace it later with the found solution
container = [u]

# This is now the function that will be used by PyMbs as a controller
# for the multibody system (input arguments set accordingly to PyMbs).
def controlForce(t, y, sensors):
    F = container[0](t)
    return F
