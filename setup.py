'''
This file is part of PyTrajectory.
'''

#! /usr/bin/env python2

from distutils.core import setup

setup(name='PyTrajectory',
    #version=pytrajectory.__version__,
    version='0.3.3',
    packages=['pytrajectory'],
    install_requires=['numpy>=1.8.1',
                    'sympy>=0.7.5',
                    'scipy>=0.13.3'],
    
    # metadata for upload to PyPI
    author='Andreas Kunze, Carsten Knoll, Oliver Schnabel',
    author_email='Andreas.Kunze@mailbox.tu-dresden.de',
    description='Python library for trajectory planning.'
    )
