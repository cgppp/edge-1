#encoding: utf-8

import torch
from torch.autograd import Function

from cnfg.ihyp import extra_compile_args, extra_cuda_compile_args

try:
	import clmvavg_cpp
except Exception as e:
	from torch.utils.cpp_extension import load
	clmvavg_cpp = load(name="clmvavg_cpp", sources=["modules/cpp/hplstm/clmvavg.cpp"], extra_cflags=extra_compile_args)
try:
	import ml2mvavgs_cuda
except Exception as e:
	import torch
	if torch.cuda.is_available():
		from torch.utils.cpp_extension import load
		ml2mvavgs_cuda = load(name="ml2mvavgs_cuda", sources=["modules/cpp/hplstm/ml2mvavgs_cuda.cpp", "modules/cpp/hplstm/ml2mvavgs_cuda_kernel.cu"], extra_cflags=extra_compile_args, extra_cuda_cflags=extra_cuda_compile_args)
	else:
		ml2mvavgs_cuda = None

class MvAvgFunction(Function):

	@staticmethod
	def forward(ctx, x, beta, inplace=False, out=None):

		if x.size(1) > 1:
			if x.is_cuda:
				_ = x.size()
				_out = (x if inplace else x.new_empty(_)) if out is None else out
				_out = ml2mvavgs_cuda.forward(x, _out, beta, *_)
			else:
				_out = (x if inplace else x.clone()) if out is None else (out if out.is_set_to(x) else out.copy_(x))
				_out = clmvavg_cpp.forward(x, _out, 1, beta)
		else:
			mbeta = 1.0 - beta
			_out = x.mul_(mbeta) if inplace and (out is None) else torch.mul(x, mbeta, out=out)
		ctx.beta = beta

		return _out

	@staticmethod
	def backward(ctx, grad_out):

		if ctx.needs_input_grad[0]:
			if grad_out.size(1) > 1:
				if grad_out.is_cuda:
					_ = grad_out.size()
					grad_x = grad_out.new_empty(_)
					grad_x = ml2mvavgs_cuda.backward(grad_out, grad_x, ctx.beta, *_)
				else:
					grad_x = clmvavg_cpp.backward(grad_out, 1, ctx.beta)
			else:
				grad_x = grad_out.mul(1.0 - ctx.beta)
			return grad_x, None, None, None
		else:
			return None, None, None, None

MvAvgFunc = MvAvgFunction.apply
