#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

from cnfg.ihyp import extra_compile_args, extra_cuda_compile_args

setup(name="lgatev_cpp", ext_modules=[cpp_extension.CppExtension("lgatev_cpp", ["modules/cpp/hplstm/lgatev.cpp"], extra_compile_args=extra_compile_args)], cmdclass={"build_ext": cpp_extension.BuildExtension})
setup(name="lgatev_nocx_cpp", ext_modules=[cpp_extension.CppExtension("lgatev_nocx_cpp", ["modules/cpp/hplstm/lgatev_nocx.cpp"], extra_compile_args=extra_compile_args)], cmdclass={"build_ext": cpp_extension.BuildExtension})
setup(name="lgates_cpp", ext_modules=[cpp_extension.CppExtension("lgates_cpp", ["modules/cpp/hplstm/lgates.cpp"], extra_compile_args=extra_compile_args + ["-fopenmp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
setup(name="lgates_cuda", ext_modules=[cpp_extension.CUDAExtension("lgates_cuda", ["modules/cpp/hplstm/lgates_cuda.cpp", "modules/cpp/hplstm/lgates_cuda_kernel.cu"], extra_compile_args={"cxx": extra_compile_args, "nvcc": extra_cuda_compile_args})], cmdclass={"build_ext": cpp_extension.BuildExtension})
setup(name="clmvavg_cpp", ext_modules=[cpp_extension.CppExtension("clmvavg_cpp", ["modules/cpp/hplstm/clmvavg.cpp"], extra_compile_args=extra_compile_args)], cmdclass={"build_ext": cpp_extension.BuildExtension})
setup(name="ml2mvavgs_cuda", ext_modules=[cpp_extension.CUDAExtension("ml2mvavgs_cuda", ["modules/cpp/hplstm/ml2mvavgs_cuda.cpp", "modules/cpp/hplstm/ml2mvavgs_cuda_kernel.cu"], extra_compile_args={"cxx": extra_compile_args, "nvcc": extra_cuda_compile_args})], cmdclass={"build_ext": cpp_extension.BuildExtension})
