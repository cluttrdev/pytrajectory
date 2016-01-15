.. _examples:

Examples
========

The following example systems from mechanics demonstrate the application 
of PyTrajectory. The derivation of the model equations is omittted here.

The source code of the examples can be downloaded `here <https://github.com/TUD-RST/pytrajectory/tree/master/dist>`_.
In order to run them simply type::

   $ python ex<ExampleNumber>_<ExampleName>.py

The results of the examples latest simulation are saved in a pickle dump file by default.
To prevent this add the *no-pickle* command line argument to the command above.

If you want to plot the results and/or animate the example system add the *plot* and/or
the *animate* argument to the command.

So the command may look something like::

   $ python ex0_InvertedPendulumSwingUp.py no-pickle plot animate

For even more examples, which might not be part of the documentation, have a look at the
`repository <https://github.com/TUD-RST/pytrajectory/tree/master/examples>`_.


.. toctree::
   :maxdepth: 1

   inv_pendulum_trans
   inv_dual_pendulum_swing
   aircraft
   uact_manipulator
   acrobot
   con_double_integrator
   con_inv_pendulum_swing
   con_double_pendulum
   inv_3_bar_pend
