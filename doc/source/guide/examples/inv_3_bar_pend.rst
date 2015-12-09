.. _inv-3-bar-pend:

Swing up of a 3-bar pendulum
----------------------------

Now we consider a cart with 3 pendulums attached to it.

To get a callable function for the vector field of this dynamical system
we need to set up and solve its motion equations for the accelaration.

Therefore, the function :py:func:`n_bar_pendulum` generates the mass matrix
:math:`M` and right hand site :math:`B` of the motion equations :math:`M\ddot{x} = B`
for a general :math:`n`\ -bar pendulum, which we use for the case :math:`n = 3`.

The formulas this function uses are taken from the project report
*'Simulation of the inverted pendulum'* written by *Christian Wachinger* and
*Michael Pock* at the *Mathematics Departement, Technical University Munich*
in December 2004.

.. only:: html
   
   .. image:: /../pic/inv_3_bar_pend.gif

Source Code
+++++++++++

.. literalinclude:: /../../examples/ex9_TriplePendulum.py
   :lines: 1-349

