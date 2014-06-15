import numpy as np

def u(t):
    return np.zeros(3)

container = [u]

def controlForce(t, y, sensors):
    F = container[0](t)
    return F
