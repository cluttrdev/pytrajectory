'''
This file is part of PyTrajectory.
'''

#! /usr/bin/env python

from setuptools import setup
import os
import codecs

import pytrajectory

here = os.path.abspath(ox.path.dirname(__file__))

def read(*parts):
    return codecs.open(os.path.join(here, *parts), 'r').read()

long_description = read('README.rst')

setup(name='PyTrajectory',
    version=pytrajectory.__version__,
    install_requires=['numpy>=1.8.1',
                    'sympy>=0.7.5',
                    'scipy>=0.13.3']
    extras_requires={'visualisation' : ['Matplotlib']},
    author='Andreas Kunze, Carsten Knoll, Oliver Schnabel',
    author_email='Andreas.Kunze@mailbox.tu-dresden.de',
    description='Python library for trajectory planning.',
    long_description=long_description,
    packages=['pytrajectory'],
    platforms='any'
    )
