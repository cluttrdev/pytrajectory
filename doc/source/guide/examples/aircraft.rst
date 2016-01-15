.. _ex_aircraft:

Aircraft
--------

In this section we consider the model of a unmanned vertical take-off 
aircraft. The aircraft has two permanently mounted thrusters on the 
wings which can apply the thrust forces :math:`F_1` and :math:`F_2` 
independently of each other. The two engines are inclined by an angle 
:math:`\alpha` with respect to the aircraft-fixed axis :math:`\eta_2` 
and engage in the points :math:`P_1 = (l, h)` and :math:`P_2 = (-l,-h)`.
The coordinates of the center of mass :math:`M` of the aircraft in the 
inertial system are denoted by :math:`z_1` and :math:`z_2`. At the same 
time, the point is the origin of the plane coordinate system. The 
aircraft axes are rotated by the angle :math:`\theta` with respect to 
the :math:`z_2`-axis.

.. image:: /../pic/aircraft.png
   :scale: 80

Through the establishment of the momentum balances for the model one
obtains the equations

.. math::
   :nowrap:

   \begin{eqnarray*}
      m \ddot{z}_1 & = & - \sin(\theta)(F_1 + F_2)\cos(\alpha) + \cos(\theta)(F_1 - F_2)\sin(\alpha) \\
      m \ddot{z}_2 & = & \cos(\theta)(F_1 + F_2)\sin(\alpha) + \sin(\theta)(F_1 - F_2)\cos(\alpha) - mg \\
      J \ddot{\theta} & = & (F_1 - F_2)(l \cos(\alpha) + h \sin(\alpha))
   \end{eqnarray*}

With the state vector :math:`x = [z_1, \dot{z}_1, z_2, \dot{z}_2, \theta, \dot{\theta}]^T`
and :math:`u = [u_1, u_2]^T = [F_1, F_2]^T` the state space 
representation of the system is as follows.

.. math::
   :nowrap:

   \begin{eqnarray*}
      \dot{x}_1 & = & x_2 \\
      \dot{x}_2 & = & \frac{1}{m}(-\sin(x_5)(u_1 + u_2)\cos(\alpha) + \cos(x_5)(u_1 - u_2)\sin(\alpha)) \\
      \dot{x}_3 & = & x_4 \\
      \dot{x}_2 & = & \frac{1}{m}(\cos(x_5)(u_1 + u_2)\cos(\alpha) + \sin(x_5)(u_1 - u_2)\sin(\alpha)) - g  \\
      \dot{x}_5 & = & x_6 \\
      \dot{x}_6 & = & \frac{1}{J}(l \cos(\alpha) + h \sin(\alpha))
   \end{eqnarray*}

For the aircraft, a trajectory should be planned that translates the 
horizontally aligned flying object from a rest position (hovering) along 
the :math:`z_1` and :math:`z_2` axis back into a hovering position. 
The hovering is to be realized on the boundary conditions of the input. 
Therefor the derivatives of the state variables should satisfy the 
following conditions. 

.. math::
   :nowrap:

   $ \dot{z}_1 = \ddot{z}_1 = \dot{z}_2 = \ddot{z_2} = \dot{\theta} = \ddot{\theta} = 0 $

For the horizontal position applies :math:`\theta = 0`. These demands 
yield the boundary conditions for the inputs.

.. math::
   :nowrap:

   $ F_1(0) = F_1(T) = F_2(0) = F_2(T) = \frac{mg}{2 \cos(\alpha)} $

.. only:: html

   .. image:: /../pic/aircraft.gif

Source Code
+++++++++++

.. literalinclude:: /../../examples/ex3_Aircraft.py
   :lines: 1-59


