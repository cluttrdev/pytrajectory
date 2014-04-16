Background
==========

This section is intended to give some insights into the mathematical 
background that is the basis of PyTrajectory.

Collocation method
------------------

Given a system of autonomous differential equations

.. math::

   \dot{x}_1(t) = f_n(x_1(t),...,x_n(t))
   
   \vdots \qquad \vdots
   
   \dot{x}_n(t) = f_n(x_1(t),...,x_n(t))

with :math:`t \in [a, b]` and *Dirichlet* boundary conditions

.. math::
   x_i(a) = \alpha_i,\quad x_i(b) = \beta_i \qquad i = 1,...,n

the collocation method to solve the problem basically works as follows.

We choose :math:`N+1` collocation points :math:`t_i,\ i = 0,...,N` from the interval 
:math:`[a, b]` where :math:`t_0 = a,\ t_{N} = b` and search for functions 
:math:`P_i:[a,b] \rightarrow R` which satisfy the following conditions:

.. math::

   P_i(t_0) = \alpha_i, \qquad P_i(t_N) = \beta_i

   \frac{d}{d t} P_i(t) = f_i(P_1(t),...,P_n(t)) \quad i = 1,...,n

Through these demands the exact solution of the ode system will be approximated.


Candidate functions
-------------------

PyTrajectory uses cubic spline functions as candidates for the approximation of the 
solution. Splines are piecewise polynomials with a global differentiability. 
The connection points :math:`\tau_i` between the polynomial sections are equidistantly 
and are referred to as nodes.

.. math::
   
   t_0 = \tau_0 < \tau_1 < ... < \tau_{\eta} = t_N \qquad h = \frac{t_N - t_0}{\eta}

   \tau_{i+1} = \tau_i + h \quad i = 0,...,\eta-1

The polynomial sections can be created as follows.

.. math::

   P_i(t) = c_{i,0}(t-i h)^3 + c_{i,1}(t-i h)^2 + c_{i,2}(t-i h) + c_{i,3} 

   c_{i,j} \in R,\qquad i = 1,...,\eta,\ j = 0,...,3

In addition to the steadiness the spline functions should be twice steadily differentiable in 
the nodes :math:`\tau`.

Equation system
---------------

... to do
