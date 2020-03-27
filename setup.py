from setuptools import setup
from Cython.Build import cythonize

setup(
    name = "Climate",
    ext_modules = cythonize(["*.py"]
                            build_dir="output"),
)