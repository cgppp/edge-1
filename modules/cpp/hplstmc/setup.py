#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

from cnfg.ihyp import extra_compile_args, extra_cuda_compile_args

setup(name="lgatec_cpp", ext_modules=[cpp_extension.CppExtension("lgatec_cpp", ["modules/cpp/hplstmc/lgate.cpp"], extra_compile_args=extra_compile_args + ["-fopenmp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
setup(name="lgatec_cuda", ext_modules=[cpp_extension.CUDAExtension("lgatec_cuda", ["modules/cpp/hplstmc/lgate_cuda.cpp", "modules/cpp/hplstmc/lgate_cuda_kernel.cu"], extra_compile_args={"cxx": extra_compile_args, "nvcc": extra_cuda_compile_args})], cmdclass={"build_ext": cpp_extension.BuildExtension})
