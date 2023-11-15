#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

from cnfg.ihyp import extra_compile_args

setup(name="self_attn_cpp", ext_modules=[cpp_extension.CppExtension("self_attn_cpp", ["modules/cpp/base/attn/self/attn.cpp"], extra_compile_args=extra_compile_args)], cmdclass={"build_ext": cpp_extension.BuildExtension})
