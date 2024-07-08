#encoding: utf-8

import torch
from torch.autograd import Function

from cnfg.ihyp import extra_compile_args, extra_cuda_compile_args

try:
	import clcumsum_cpp
except Exception as e:
	from torch.utils.cpp_extension import load
	clcumsum_cpp = load(name="clcumsum_cpp", sources=["utils/cpp/hplstm/clcumsum.cpp"], extra_cflags=extra_compile_args)
try:
	import ml2cumsums_cuda
except Exception as e:
	import torch
	if torch.cuda.is_available():
		from torch.utils.cpp_extension import load
		ml2cumsums_cuda = load(name="ml2cumsums_cuda", sources=["utils/cpp/hplstm/ml2cumsums_cuda.cpp", "utils/cpp/hplstm/ml2cumsums_cuda_kernel.cu"], extra_cflags=extra_compile_args, extra_cuda_cflags=extra_cuda_compile_args)
	else:
		ml2cumsums_cuda = None

class cumsumFunction(Function):

	@staticmethod
	def forward(ctx, x, inplace=False):

		if x.size(1) > 1:
			if x.is_cuda:
				_ = x.size()
				_out = ml2cumsums_cuda.forward(x, x if inplace else x.new_empty(_), *_)
			else:
				_out = clcumsum_cpp.forward(x if inplace else x.clone(), 1)
		else:
			_out = x if inplace else x.clone()

		return _out

	@staticmethod
	def backward(ctx, grad_out):

		if ctx.needs_input_grad[0]:
			if grad_out.size(1) > 1:
				if grad_out.is_cuda:
					_ = grad_out.size()
					grad_x = ml2cumsums_cuda.backward(grad_out, grad_out.new_empty(_), *_)
				else:
					grad_x = clcumsum_cpp.backward(grad_out.clone(), 1)
			else:
				grad_x = grad_out
			return grad_x, None
		else:
			return None, None

cumsumFunc = cumsumFunction.apply
