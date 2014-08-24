.. _ex_acrobot:

Acrobot
-------

One further interesting example is that of the acrobot. The model can be 
regarded as a simplified gymnast hanging on a horizontal bar with both hands. 
The movements of the entire system is to be controlled only by movement of the hip. 
The body of the gymnast is represented by two rods which are jointed in the joint
:math:`G_2`. The first rod is movably connected at joint :math:`G_1` with the inertial 
system, which corresponds to the encompassing of the stretching rod with the hands.

For the model, two equal-length rods with a length :math:`l_1 = l_2 = l` are assumed 
with a homogeneous distribution of mass :math:`m_1 = m_2 = m` over the entire rod length.
This does not correspond to the proportions of a man, also no restrictions were placed 
on the mobility of the hip joint. 

The following figure shows the schematic representation of the model.

.. image:: /../pic/acrobot.png
   :scale: 80

Using the previously assumed model parameters and the write abbreviations

.. math::
   :nowrap:

   \begin{eqnarray*}
      I      & = & \frac{1}{3}m l^2 \\
      d_{11} & = & \frac{m l^2}{4} + m(l^2 + \frac{l^2}{4} + l^2 \cos(\theta_2)) + 2I \\
      h_1    & = & - \frac{m l^2}{2} \sin(\theta_2) (\dot{\theta}_2 (\dot{\theta}_2 + 2\dot{\theta}_1)) \\
      d_{12} & = & m (\frac{l^2}{4} + \frac{l^2}{2} \cos(\theta_1)) + I \\
      \varphi_1 & = & \frac{3}{2}m l g \cos(\theta_1) + \frac{1}{2}m l g \cos(\theta_1 + \theta_2)
    \end{eqnarray*}

as well as the state vector :math:`x = [\theta_2, \dot{\theta}_2, \theta_1, \dot{\theta}_1]` one obtains
the following state representation with the virtual input :math:`u = \ddot{\theta}_2`

.. math::
   :nowrap:

   \begin{eqnarray*}
      \dot{x}_1 & = & x_2 \\
      \dot{x}_2 & = & u \\
      \dot{x}_3 & = & x_4 \\
      \dot{x}_4 & = & -d_{11}^{-1} (h_1 + \varphi_1 + d_{12}u)
   \end{eqnarray*}

Now, the trajectory of the manipulated variable for an oscillation of the gymnast should be determined.
The starting point of the exercise are the two downward hanging rods. These are to be transferred into another 
rest position in which the two bars show vertically upward within an operating time of :math:`T = 2 [s]`. 
At the beginning and end of the process, the input variable is to merge continuously into the rest 
position :math:`u(0) = u(T) = 0`.

The initial and final states thus are

.. math::
   :nowrap:

   \begin{equation*}
      x(0) = \begin{bmatrix} 0 \\ 0 \\ \frac{3}{2} \pi \\ 0 \end{bmatrix}
      \rightarrow
      x(T) = \begin{bmatrix} 0 \\ 0 \\ \frac{1}{2} \pi \\ 0 \end{bmatrix}
   \end{equation*}


Source Code
+++++++++++

.. literalinclude:: /../../examples/ex5_Acrobot.py
   :lines: 1-58

.. only:: html

   .. image:: /../pic/acrobot.gif

