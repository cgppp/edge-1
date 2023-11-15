#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

from cnfg.ihyp import extra_compile_args, extra_cuda_compile_args

setup(name="idweightacc_cpp", ext_modules=[cpp_extension.CppExtension("idweightacc_cpp", ["modules/cpp/eqsparse/idweightacc.cpp"], extra_compile_args=extra_compile_args + ["-fopenmp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
setup(name="idweightacc_bias_cpp", ext_modules=[cpp_extension.CppExtension("idweightacc_bias_cpp", ["modules/cpp/eqsparse/idweightacc_bias.cpp"], extra_compile_args=extra_compile_args + ["-fopenmp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
setup(name="idweightacc_cuda", ext_modules=[cpp_extension.CUDAExtension("idweightacc_cuda", ["modules/cpp/eqsparse/idweightacc_cuda.cpp", "modules/cpp/eqsparse/idweightacc_cuda_kernel.cu"], extra_compile_args={"cxx": extra_compile_args, "nvcc": extra_cuda_compile_args})], cmdclass={"build_ext": cpp_extension.BuildExtension})
setup(name="idweightacc_bias_cuda", ext_modules=[cpp_extension.CUDAExtension("idweightacc_bias_cuda", ["modules/cpp/eqsparse/idweightacc_bias_cuda.cpp", "modules/cpp/eqsparse/idweightacc_bias_cuda_kernel.cu"], extra_compile_args={"cxx": extra_compile_args, "nvcc": extra_cuda_compile_args})], cmdclass={"build_ext": cpp_extension.BuildExtension})
