from distutils.core import setup, Extension
import numpy

ext = Extension('core',
                sources=['core.c'],
                include_dirs=[numpy.get_include()])

setup(name='monotonic_align', ext_modules=[ext])
