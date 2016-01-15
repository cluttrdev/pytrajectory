.. _ex_unact_mani:

Underactuated Manipulator
-------------------------

In this section, the model of an underactuated manipulator is treated. 
The system consists of two bars with the mass :math:`M_1` and 
:math:`M_2` which are connected to each other via the joint :math:`G_2`.
The angle between them is designated by :math:`\theta_2`. The joint 
:math:`G_1` connects the first rod with the inertial system, the angle 
to the :math:`x`-axis is labeled :math:`\theta_1`.
In the joint :math:`G_1` the actuating torque :math:`Q` is applied. The 
bars have the moments of inertia :math:`I_1` and :math:`I_2`. The 
distances between the centers of mass to the joints are :math:`r_1` and 
:math:`r_2`. 

.. image:: /../pic/uact_manipulator.png
   :scale: 80

The modeling was taken from the thesis of Carsten Knoll 
(April, 2009) where in addition the inertia parameter :math:`\eta` was 
introduced.

.. math::
   :nowrap:

   \begin{equation*}
      \eta = \frac{m_2 l_1 r_2}{I_2 + m_2 r_2^2}
   \end{equation*}

For the example shown here, strong inertia coupling was assumed with 
:math:`\eta = 0.9`. By partial linearization to the output :math:`y = 
\theta_1` one obtains the state representation with the states 
:math:`x = [\theta_1, \dot{\theta}_1, \theta_2, \dot{\theta}_2]^T` and 
the new input :math:`\tilde{u} = \ddot{\theta}_1`.

.. math::
   :nowrap:

   \begin{eqnarray*}
      \dot{x}_1 & = & x_2 \\
      \dot{x}_2 & = & \tilde{u} \\
      \dot{x}_3 & = & x_4 \\
      \dot{x}_4 & = & -\eta x_2^2  \sin(x_3) - (1 + \eta \cos(x_3))\tilde{u}
   \end{eqnarray*}

For the system, a trajectory is to be determined for the transfer 
between two equilibrium positions within an operating time of 
:math:`T = 1.8 [s]`. 

.. math::
   :nowrap:
   
   \begin{equation*}
      x(0) = \begin{bmatrix} 0 \\ 0 \\ 0.4 \pi \\ 0 \end{bmatrix}
      \rightarrow
      x(T) = \begin{bmatrix} 0.2 \pi \\ 0 \\ 0.2 \pi \\ 0 \end{bmatrix}
   \end{equation*}

The trajectory of the inputs should be without cracks in the transition 
to the equilibrium positions (:math:`\tilde{u}(0) = \tilde{u}(T) = 0`).

.. only:: html

   .. image:: /../pic/uact_manipulator.gif

Source Code
+++++++++++

.. literalinclude:: /../../examples/ex4_UnderactuatedManipulator.py
   :lines: 1-50

