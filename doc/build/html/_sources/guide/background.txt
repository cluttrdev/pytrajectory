Background
==========

This section is intended to give some insights into the mathematical 
background that is the basis of PyTrajectory.

Collocation method
------------------

Given a system of autonomous differential equations

.. math::

   \dot{x}_1(t) = f_1(x_1(t),...,x_n(t))
   
   \vdots \qquad \vdots
   
   \dot{x}_n(t) = f_n(x_1(t),...,x_n(t))

with :math:`t \in [a, b]` and *Dirichlet* boundary conditions

.. math::
   x_i(a) = \alpha_i,\quad x_i(b) = \beta_i \qquad i = 1,...,n

the collocation method to solve the problem basically works as follows.

We choose :math:`N+1` collocation points :math:`t_j,\ j = 0,...,N` from the interval 
:math:`[a, b]` where :math:`t_0 = a,\ t_{N} = b` and search for functions 
:math:`P_i:[a,b] \rightarrow \mathbb{R}` which for all :math:`j = 0,..,N` satisfy the 
following conditions:

.. math::
   :nowrap:

   \begin{equation}
      P_i(t_0) = \alpha_i, \qquad P_i(t_N) = \beta_i
   \end{equation}
   
   \begin{equation}
      \frac{d}{d t} P_i(t_j) = f_i(P_1(t_j),...,P_n(t_j)) \quad i = 1,...,n
   \end{equation}

Through these demands the exact solution of the ode system will be approximated.
The demands on the boundary values :math:`(1)` ​​can be sure already by suitable 
construction of the shape functions. This results in the following system of equations.

.. math::

   \frac{d}{d t}P_1(t_0) - f(P_1(t_0)) := G_1^0(c) = 0

   \qquad \vdots

   \frac{d}{d t}P_n(t_0) - f(P_n(t_0)) := G_n^0(c) = 0

   \qquad \vdots

   \frac{d}{d t}P_1(t_1) - f(P_1(t_1)) := G_1^1(c) = 0

   \qquad \vdots

   \frac{d}{d t}P_n(t_N) - f(P_n(t_N)) := G_n^N(c) = 0

Solving the boundary value problem is thus reduced to the finding of a zero point 
of :math:`G = (G_1^0 ,..., G_n^N)^T`, where :math:`c` is the vector of all independent
parameters that result from the candidate functions.


Candidate functions
-------------------

PyTrajectory uses cubic spline functions as candidates for the approximation of the 
solution. Splines are piecewise polynomials with a global differentiability. 
The connection points :math:`\tau_k` between the polynomial sections are equidistantly 
and are referred to as nodes.

.. math::
   
   t_0 = \tau_0 < \tau_1 < ... < \tau_{\eta} = t_N \qquad h = \frac{t_N - t_0}{\eta}

   \tau_{k+1} = \tau_k + h \quad k = 0,...,\eta-1

The :math:`\eta` polynomial sections can be created as follows.

.. math::

   S_k(t) = c_{k,0}(t-k h)^3 + c_{k,1}(t-k h)^2 + c_{k,2}(t-k h) + c_{k,3} 

   c_{k,l} \in \mathbb{R},\qquad k = 1,...,\eta,\ l = 0,...,3

Then, each spline function is defined by

.. math::
   :nowrap:

   \begin{equation*}
      P_i(t) = 
      \begin{cases}
         S_1(t) & t_0 \leq t < h \\
         \vdots & \vdots \\
         S_k(t) & (k-1)h \leq t < k h \\
         \vdots & \vdots \\
         S_\eta(t) & (\eta-1)h \leq t \leq \eta h
      \end{cases}
   \end{equation*}

In addition to the steadiness the spline functions should be twice steadily differentiable in 
the nodes :math:`\tau`. Therefor, three smoothness conditions can be set up in all 
:math:`\tau_k, k = 1,...,\eta-1`.

.. math::
   :nowrap:

   \begin{eqnarray*}
      S_k(k h) & = & S_{k+1}(k h) \\
      \frac{d}{d t} S_k(k h) & = & \frac{d}{d t} S_{k+1}(k h) \\
      \frac{d^2}{d t^2} S_k(k h) & = & \frac{d^2}{d t^2} S_{k+1}(k h)
   \end{eqnarray*}


Furthermore, conditions can be set up at the edges arising from the boundary conditions of 
the differential equation system.

.. math::
   :nowrap:

   \begin{equation*}
      \frac{d^j}{d t^j} S_1(\tau_0) = \alpha_j \qquad \frac{d^j}{d t^j} S_\eta(\tau_\eta) = \beta_j \qquad j = 0,...,\nu
   \end{equation*}

The degree :math:`\nu` of the boundary conditions depends on the structure of the differential
equation system.
