from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [ Extension("simulation", ["simulation.py"]),
                Extension("solver", ["solver.py"]),
                Extension("spline", ["spline.py"]),
                Extension("trajectory", ["trajectory.py"]),
                Extension("tools", ["tools.py"])]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
    )