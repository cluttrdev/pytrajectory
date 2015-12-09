.. _ex_inv_pend:

Translation of the inverted pendulum
------------------------------------

An example often used in literature is the inverted pendulum. Here a 
force :math:`F` acts on a cart with mass :math:`M_w`. In addition the 
cart is connected by a massless rod with a pendulum mass :math:`m_p`.
The mass of the pendulum is concentrated in :math:`P_2` and that of the 
cart in :math:`P_1`. The state vector of the system can be specified 
using the carts position :math:`x_w(t)` and the pendulum deflection 
:math:`\varphi(t)` and their derivatives. 

.. image:: /../pic/inv_pendulum.png
   :scale: 80

With the *Lagrangian Formalism* the model has the following state 
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

.. only:: html

   .. image:: /../pic/inv_pend_trans.gif

Source Code
+++++++++++

.. literalinclude:: /../../examples/ex1_InvertedPendulumTranslation.py
   :lines: 1-48

