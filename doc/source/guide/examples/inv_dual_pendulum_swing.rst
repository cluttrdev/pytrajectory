.. _ex_inv_dbl_pend:

Swing up of the inverted dual pendulum
--------------------------------------

In this example we add another pendulum to the cart in the system.

.. image:: /../pic/inv_dual_pendulum.png
   :scale: 80

The system has the state vector :math:`x = [x_1, \dot{x}_1, 
\varphi_1, \dot{\varphi}_1, \varphi_2, \dot{\varphi}_2]`. A partial 
linearization with :math:`y = x_1` yields the following system state 
representation where :math:`\tilde{u} = \ddot{y}`. 

.. math::
   :nowrap:

   \begin{eqnarray*}
      \dot{x}_1 & = & x_2 \\
      \dot{x}_2 & = & \tilde{u} \\
      \dot{x}_3 & = & x_4 \\
      \dot{x}_4 & = & \frac{1}{l_1}(g \sin(x_3) + \tilde{u} \cos(x_3)) \\
      \dot{x}_5 & = & x_6 \\
      \dot{x}_6 & = & \frac{1}{l_2}(g \sin(x_5) + \tilde{u} \cos(x_5))
   \end{eqnarray*}

Here a trajectory should be planned that transfers the system between 
the following two positions of rest. At the beginning both pendulums 
should be directed downwards (:math:`\varphi_1 = \varphi_2 = \pi`).
After a operating time of :math:`T = 2 [s]` the cart should be at the 
same position again and the pendulums should be at rest with 
:math:`\varphi_1 = \varphi_2 = 0`.

.. math::
   :nowrap:

   \begin{equation*}
      x(0) = \begin{bmatrix} 0 \\ 0 \\ \pi \\ 0 \\ \pi \\ 0 \end{bmatrix} 
      \rightarrow
      x(T) = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}
   \end{equation*}

Source Code
+++++++++++

.. literalinclude:: /../../examples/ex2_InvertedDualPendulumSwingUp.py
   :lines: 1-46

.. only:: html

   .. image:: /../pic/inv_dual_pend_swing.gif

