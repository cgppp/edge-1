#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

from cnfg.ihyp import extra_compile_args

setup(name="lgate_cpp", ext_modules=[cpp_extension.CppExtension("lgate_cpp", ["modules/cpp/hplstm/lgate.cpp"], extra_compile_args=extra_compile_args)], cmdclass={"build_ext": cpp_extension.BuildExtension})
setup(name="lgate_nocx_cpp", ext_modules=[cpp_extension.CppExtension("lgate_nocx_cpp", ["modules/cpp/hplstm/lgate_nocx.cpp"], extra_compile_args=extra_compile_args)], cmdclass={"build_ext": cpp_extension.BuildExtension})
