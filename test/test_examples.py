# IMPORTS

import os
import sys
import inspect
import pytest


import pytrajectory

class TestPath(object):
    
    def test_example_path(self):
        #pth = pytrajectory.__path__[0].split(os.sep)[:-1] + ['examples']
        #examples_dir = os.sep.join(pth)
        
        pth = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        pth = pth.split(os.sep)[:-1]
        examples_dir = os.sep.join(pth + ['examples'])
        
        print pth
        print examples_dir
        
        script = os.path.join(examples_dir, 'ex1_InvertedPendulumTranslation.py')
        f = open(script)
        f.close()

class ATestExamples(object):

    pth = pytrajectory.__path__[0].split(os.sep)[:-1] + ['examples']
    examples_dir = os.sep.join(pth)

    def assert_reached_accuracy(self, loc):
        for value in loc.values():
            if isinstance(value, pytrajectory.system.ControlSystem):
                assert value.reached_accuracy

    def test_inverted_pendulum_translation(self):
        script = os.path.join(self.examples_dir, 'ex1_InvertedPendulumTranslation.py')
        d = dict(locals(), **globals())
        execfile(script, d, d)
        self.assert_reached_accuracy(locals())

    def test_inverted_dual_pendulum_swing_up(self):
        script = os.path.join(self.examples_dir, 'ex2_InvertedDualPendulumSwingUp.py')
        d = dict(locals(), **globals())
        execfile(script, d, d)
        self.assert_reached_accuracy(locals())

    def test_aricraft(self):
        script = os.path.join(self.examples_dir, 'ex3_Aircraft.py')
        d = dict(locals(), **globals())
        execfile(script, d, d)
        self.assert_reached_accuracy(locals())
    
    @pytest.mark.slow
    def test_underactuated_manipulator(self):
        script = os.path.join(self.examples_dir, 'ex4_UnderactuatedManipulator.py')
        d = dict(locals(), **globals())
        execfile(script, d, d)
        self.assert_reached_accuracy(locals())
    
    @pytest.mark.slow
    def test_acrobot(self):
        script = os.path.join(self.examples_dir, 'ex5_Acrobot.py')
        d = dict(locals(), **globals())
        execfile(script, d, d)
        self.assert_reached_accuracy(locals())
    
    @pytest.mark.slow
    def test_constrained_double_integrator(self):
        script = os.path.join(self.examples_dir, 'ex6_ConstrainedDoubleIntegrator.py')
        d = dict(locals(), **globals())
        execfile(script, d, d)
        self.assert_reached_accuracy(locals())

    @pytest.mark.slow
    def test_constrained_inverted_pendulum(self):
        script = os.path.join(self.examples_dir, 'ex7_ConstrainedInvertedPendulum.py')
        d = dict(locals(), **globals())
        execfile(script, d, d)
        self.assert_reached_accuracy(locals())

    @pytest.mark.slow
    def test_constrained_double_pendulum(self):
        script = os.path.join(self.examples_dir, 'ex8_ConstrainedDoublePendulum.py')
        d = dict(locals(), **globals())
        execfile(script, d, d)
        self.assert_reached_accuracy(locals())
