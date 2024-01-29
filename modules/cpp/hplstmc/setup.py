#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

from cnfg.ihyp import extra_compile_args

setup(name="lgatec_cpp", ext_modules=[cpp_extension.CppExtension("lgatec_cpp", ["modules/cpp/hplstmc/lgate.cpp"], extra_compile_args=extra_compile_args + ["-fopenmp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
