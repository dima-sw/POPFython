from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "predict",
        ["predict.pyx"],
        extra_compile_args=['-ffast-math','-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='predict-popf',
    ext_modules=cythonize(ext_modules),
)