#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

from cnfg.ihyp import extra_compile_args

setup(name="spreader_cpp", ext_modules=[cpp_extension.CppExtension("spreader_cpp", ["modules/cpp/spreader/spreader.cpp"], extra_compile_args=extra_compile_args)], cmdclass={"build_ext": cpp_extension.BuildExtension})
setup(name="spreader_nocx_cpp", ext_modules=[cpp_extension.CppExtension("spreader_nocx_cpp", ["modules/cpp/spreader/spreader_nocx.cpp"], extra_compile_args=extra_compile_args)], cmdclass={"build_ext": cpp_extension.BuildExtension})
