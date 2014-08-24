'''
This file is part of PyTrajectory.
'''

from distutils.core import setup

setup(name='PyTrajectory',
    version='0.3.4',
    packages=['pytrajectory'],
    requires=['numpy (>=1.8.1)',
                'sympy (>=0.7.5)',
                'scipy (>=0.13.3)',
                'matplotlib'],
    
    # metadata
    author='Andreas Kunze, Carsten Knoll, Oliver Schnabel',
    author_email='Andreas.Kunze@mailbox.tu-dresden.de',
    url='https://bitbucket.org/akunze3/pytrajectory',
    description='Python library for trajectory planning.',
    long_description='''
    PyTrajectory is a Python library for the determination of the feed forward 
    control to achieve a transition between desired states of a nonlinear control system.
    '''
    )
