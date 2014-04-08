Translation of the inverse pendulum
-----------------------------------

An example often used in literature is the inverse pendulum. Here a 
force :math:`F` acts on a cart with mass :math:`M_w`. In addition the 
cart is connected by a massless rod with a pendulum mass :math:`m_p`.
The mass of the pendulum is concentrated in :math:`P_2` and that of the 
cart in :math:`P_1`. The state vector of the system can be specified 
using the carts position :math:`x_w(t)` and the pendulum deflection 
:math:`\varphi(t)` and their derivatives. 

.. image:: /../../pic/inv_pendulum.png

With the *Lagrangian Formalism* the model has the following state space 
representation where :math:`u_1 = F` and 
:math:`x = [x_1, x_2, x_3, x_4] = [x_w, \dot{x}_w, \varphi, \dot{\varphi}]`

.. math::
   :nowrap:
   
   \begin{eqnarray*}
      \dot{x}_1 & = & x_2 \\
      \dot{x}_2 & = & \frac{m_p \sin(x_3)(-l x_4^2 + g \cos x_3)}{M_w l + m_p \sin^2(x_3)} + \frac{\cos(x_3)}{M_w l + m_p l \sin^2(x_3)} u_1 \\
      \dot{x}_3 & = & x_4 \\
      \dot{x}_4 & = & \frac{\sin(x_3)(-m_p l x_4^2 \cos(x_3) + g(M_w + m_p))}{M_w l + m_p \sin^2(x_3)} + \frac{\cos(x_3)}{M_w l + m_p l \sin^2(x_3)} u_1
   \end{eqnarray*}

A possibly wanted trajectory is the translation of the cart along the 
x-axis (i.e. by :math:`0.5m`). In the beginning and end of the process 
the cart and pendulum should remain at rest and the pendulum should be 
aligned vertically upwards (:math:`\varphi = 0`). As a further condition 
:math:`u_1` should start and end steadily in the rest position 
(:math:`u_1(0) = u_1(T) = 0`).
The operating time here is :math:`T = 1 [s]`.

.. code-block:: python

   # import trajectory class and necessary dependencies
   from pytrajectory.trajectory import Trajectory
   from sympy import sin, cos
   import numpy as np
   
   # define the function that returns the vectorfield
   def f(x,u):
       x1, x2, x3, x4 = x	# system state variables
       u1, = u			# input variable
       
       l = 0.5     # length of the pendulum rod
       g = 9.81    # gravitational acceleration
       M = 1.0     # mass of the cart
       m = 0.1     # mass of the pendulum
       
       s = sin(x3)
       c = cos(x3)
       
       ff = np.array([                     x2,
                      m*s*(-l*x4**2+g*c)/(M+m*s**2)+1/(M+m*s**2)*u1,
                                           x4,
               s*(-m*l*x4**2*c+g*(M+m))/(M*l+m*l*s**2)+c/(M*l+l*m*s**2)*u1
                   ])
       return ff
   
   # boundary values at the start (a = 0.0 [s])
   xa = [  0.0,
           0.0,
           0.0,
           0.0]
   
   # boundary values at the end (b = 1.0 [s])
   xb = [  0.5,
           0.0,
           0.0,
           0.0]
   
   # create trajectory object
   T = Trajectory(f, a=0.0, b=1.0, xa=xa, xb=xb)
   
   # run iteration
   T.startIteration()
   
   # show results
   T.plot()
