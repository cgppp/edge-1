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
	if torch.cuda.is_available():
		from torch.utils.cpp_extension import load
		ml2cumsums_cuda = load(name="ml2cumsums_cuda", sources=["utils/cpp/hplstm/ml2cumsums_cuda.cpp", "utils/cpp/hplstm/ml2cumsums_cuda_kernel.cu"], extra_cflags=extra_compile_args, extra_cuda_cflags=extra_cuda_compile_args)
	else:
		ml2cumsums_cuda = None

# memory friendly implementation of: torch.cat((x.new_zeros(bsize, 1, nheads, adim), x.narrow(1, 0, seql - 1).cumsum(1),), dim=1)
class RS1cumsumFunction(Function):

	@staticmethod
	def forward(ctx, x):

		bsize, seqlen, nhead, isize = x.size()
		if seqlen > 1:
			_out = x.new_empty(bsize, seqlen, nhead, isize)
			_out.select(1, 0).zero_()
			_ = seqlen - 1
			torch.cumsum(x.narrow(1, 0, _), dim=1, out=_out.narrow(1, 1, _))
		else:
			_out = x.new_zeros(1, 1, 1, 1).expand(bsize, seqlen, nhead, isize)

		return _out

	@staticmethod
	def backward(ctx, grad_out):

		if ctx.needs_input_grad[0]:
			bsize, seqlen, nhead, isize = grad_out.size()
			if seqlen > 1:
				grad_x = grad_out.new_empty(bsize, seqlen, nhead, isize)
				grad_x.select(1, -1).zero_()
				_ = seqlen - 1
				if _ > 1:
					if grad_out.is_cuda:
						ml2cumsums_cuda.backward(grad_out.narrow(1, 1, _), grad_x.narrow(1, 0, _), bsize, _, nhead, isize)
					else:
						clcumsum_cpp.backward(grad_x.narrow(1, 0, _).copy_(grad_out.narrow(1, 1, _)), 1)
				else:
					grad_x.narrow(1, 0, _).copy_(grad_out.narrow(1, 1, _))

				return grad_x

		return None

RS1cumsumFunc = RS1cumsumFunction.apply
