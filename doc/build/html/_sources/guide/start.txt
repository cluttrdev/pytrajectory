Getting Started
***************

This section provides an overview on what PyTrajectory is and how to use it.
For a more detailed view please have a look at the :ref:`pytrajectory`.

.. contents:: Contents
   :local:


What is PyTrajectory?
=====================

PyTrajectory is a Python library for the determination of the feed forward control 
to achieve a transition between desired states of a nonlinear control system.

Planning and designing of trajectories represents an important task in 
the control of technological processes. Here the problem is attributed 
on a multi-dimensional boundary value problem with free parameters.
In general this problem can not be solved analytically. It is therefore 
resorted to the method of collocation in order to obtain a numerical 
approximation.

PyTrajectory allows a flexible implementation of various tasks and enables an easy 
implementation. It suffices to supply a function :math:`f(x,u)` that represents the 
vectorfield of a control system and to specify the desired boundary values.


Installation
============

PyTrajectory was developed and tested on Python 2.7

Dependencies
------------

Before you install PyTrajectory make sure you have the following 
dependencies installed on your system.

* numpy
* sympy
* scipy
* optional
   * matplotlib [visualisation]
   * ipython [debugging]

To install PyTrajectory, simply: ::

   pip install pytrajectory

or download the tarball from here (insert link!) extract it and run ::

   python setup.py install

from within the extracted folder.


Usage
=====

... to do


A First Example
===============

... to do

