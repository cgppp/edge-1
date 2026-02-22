#encoding: utf-8

from torch.autograd import Function

from cnfg.ihyp import extra_compile_args, extra_cuda_compile_args

try:
	import clmvavgis_cpp
except Exception as e:
	from torch.utils.cpp_extension import load
	clmvavgis_cpp = load(name="clmvavgis_cpp", sources=["utils/cpp/hplstm/clmvavgis.cpp"], extra_cflags=extra_compile_args)
try:
	import ml2mvavgis_cuda
except Exception as e:
	import torch
	if torch.cuda.is_available():
		from torch.utils.cpp_extension import load
		ml2mvavgis_cuda = load(name="ml2mvavgis_cuda", sources=["utils/cpp/hplstm/ml2mvavgs_cuda.cpp", "utils/cpp/hplstm/ml2mvavgis_cuda_kernel.cu"], extra_cflags=extra_compile_args, extra_cuda_cflags=extra_cuda_compile_args)
	else:
		ml2mvavgis_cuda = None

class MvAvgiSFunction(Function):

	@staticmethod
	def forward(ctx, x, beta, inplace=False):

		if x.size(1) > 1:
			if x.is_cuda:
				_ = x.size()
				_out = ml2mvavgis_cuda.forward(x, x if inplace else x.new_empty(_), beta, *_)
			else:
				_out = clmvavgis_cpp.forward(x if inplace else x.clone(), 1, beta)
		else:
			_out = x
		ctx.beta = beta

		return _out

	@staticmethod
	def backward(ctx, grad_out):

		if ctx.needs_input_grad[0]:
			if grad_out.size(1) > 1:
				if grad_out.is_cuda:
					_ = grad_out.size()
					grad_x = ml2mvavgis_cuda.backward(grad_out, grad_out.new_empty(_), ctx.beta, *_)
				else:
					grad_x = clmvavgis_cpp.backward(grad_out.clone(), 1, ctx.beta)
			else:
				grad_x = grad_out
			return grad_x, None, None
		else:
			return None, None, None

MvAvgiSFunc = MvAvgiSFunction.apply
