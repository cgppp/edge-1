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

# memory friendly implementation of: _ = torch.cat((x.new_zeros(bsize, 1, nheads, adim), x.cumsum(1),), dim=1), (_.narrow(1, 0, seqlen), _.narrow(1, seqlen, 1))
class RS1cumsumstatFunction(Function):

	@staticmethod
	def forward(ctx, x):

		bsize, seqlen, nhead, isize = x.size()
		if seqlen > 1:
			_mem = x.new_empty(bsize, seqlen + 1, nhead, isize)
			_mem.select(1, 0).zero_()
			torch.cumsum(x, dim=1, out=_mem.narrow(1, 1, seqlen))
			_out, _state = _mem.narrow(1, 0, seqlen), _mem.narrow(1, seqlen, 1)
		else:
			_out, _state = x.new_zeros(1, 1, 1, 1).expand(bsize, seqlen, nhead, isize), x

		return _out, _state

	@staticmethod
	def backward(ctx, grad_out, grad_state):

		if ctx.needs_input_grad[0]:
			bsize, seqlen, nhead, isize = grad_out.size()
			if seqlen > 1:
				grad_x = grad_out.new_empty(bsize, seqlen, nhead, isize)
				_ = seqlen - 1
				grad_x.narrow(1, _, 1).copy_(grad_state)
				grad_x.narrow(1, 0, _).copy_(grad_out.narrow(1, 1, _))
				if _ > 1:
					if grad_out.is_cuda:
						ml2cumsums_cuda.backward(grad_x, grad_x, bsize, seqlen, nhead, isize)
					else:
						clcumsum_cpp.backward(grad_x, 1)
				else:
					grad_x.narrow(1, 0, _).copy_(grad_out.narrow(1, 1, _))

				return grad_x
			return grad_state

		return None

RS1cumsumstatFunc = RS1cumsumstatFunction.apply
