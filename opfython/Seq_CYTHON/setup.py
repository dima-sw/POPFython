from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules= cythonize('find_prot.pyx'))
setup(ext_modules= cythonize('fit.pyx'))
setup(ext_modules= cythonize('predict.pyx'))
