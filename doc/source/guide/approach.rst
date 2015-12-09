.. _non-standard-approach:

Non-Standard Approach
=====================

There are implemented two different approaches for the polynomial parts
of the splines in PyTrajectory. They differ in the choice of the nodes.
Given an interval :math:`[x_k, x_{k+1}],\ k \in [0, \ldots, N-1]` the standard
way in spline interpolation is to define the corresponding polynomial
part using the left endpoint by

.. math::

   P_k(t) = c_{k,0} (t - x_k)^3 + c_{k,1}(t - x_k)^2 + c_{k,2}(t - x_k) + c_{k,3}

However, in the thesis of O. Schnabel from which PyTrajectory emerged a different approach
is used. He defined the polynomial parts using the right endpoint, i.e.

.. math::

   P_k(t) = c_{k,0} (t - x_{k+1})^3 + c_{k,1}(t - x_{k+1})^2 + c_{k,2}(t - x_{k+1}) + c_{k,3}

This results in a different matrix to ensure the smoothness and boundary conditions
of the splines:

.. math::
   :nowrap:
   
   \setcounter{MaxMatrixCols}{20}
   \setlength{\arraycolsep}{3pt}
   \newcommand\bigzero{\makebox(0,0){\text{\huge0}}}
   \begin{equation*}
   \textstyle
   \underbrace{\begin{bmatrix}
         0 & 0   & 0  & 1 &  h^3  & -h^2   &  h & -1 \\
         0 & 0   & 1  & 0 & -3h^2 &  2h    & -1 &  0  &&&& \bigzero \\
         0 & 2   & 0  & 0 &   6h  &  -2    &  0 &  0 \\
           &     &    &   &   0   &   0    &  0 &  1  &  h^3  & -h^2 &  h & -1 \\
           &  \bigzero   &    &   &   0   &   0    &  1 &  0  & -3h^2 &  2h  & -1 &  0 &&&&&& \bigzero \\
           &     &    &   &   0   &   2    &  0 &  0  &   6h  &  -2  &  0 &  0 \\
           &&&&&&&&&&& \ddots \\ 
           &     &    &   &       &        &    &     &       &      &    &    & 0 & 0 & 0 & 1 &  h^3  & -h^2 &  h & -1 \\
           &     &    &   &       &        &  \bigzero  &     &       &      &    &    & 0 & 0 & 1 & 0 & -3h^2 &  2h  & -1 &  0 \\
           &     &    &   &       &        &    &     &       &      &    &    & 0 & 2 & 0 & 0 &   6h  &  -2  &  0 &  0 \\
           &     &    &   &       &        &    &     &       &      &    &    &   & \\
      -h^3 & h^2 & -h & 1 \\
      3h^2 & -2h &  1 & 0 &&&&&&&& \bigzero \\
      -6h  &  2  &  0 & 0 \\
           &     &    &   &       &        &    &     &       &      &    &    &   &   &   &   &   0   &    0 &  0 &  1 \\
           &     &    &   &       &        &  \bigzero  &     &       &      &    &    &   &   &   &   &   0   &    0 &  1 &  0 \\
           &     &    &   &       &        &    &     &       &      &    &    &   &   &   &   &   0   &    2 &  0 &  0 \\
   \end{bmatrix}}_{=: \boldsymbol{M}}
   \end{equation*}

..
   Another difference between the two approaches arises when computing a new initial guess
   for the free parameters after the number of spline parts has been raised.
   When using the standard approach, we can simply interpolate the splines from the last
   iteration. Though, when using the non-standard approach by now we set up an equation
   system for the free parameters by evaluating the splines from the last iteration in some
   points and demand that the newly created splines coincide with them in those points.

   The resulting equation systems matrix, however, is singular if we include the endpoints of the
   considered time interval, because the newly created splines as well as the `old` ones
   already take the same values there. But if we don't include them it turned out that
   the splines using the solution of the equation system strongly oscillate at the endpoints.
   To get rid of these oscillations we included the endpoints an solved the equation system
   as a least squares problem. By that, the oscillations are much smaller but the new splines
   don't interpolate the old ones exactly like in the standard approach.

The reason for the two different approaches being implemented is that after implementing and switching to the standard approach
some of the examples no longer converged to a solution.
The examples that are effected by that are:

- :ref:`con-double-pendulum`
- :ref:`inv-3-bar-pend`
