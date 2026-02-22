#encoding: utf-8

import torch
from torch.autograd import Function

from cnfg.ihyp import extra_compile_args, extra_cuda_compile_args

try:
	import clmvavg_cpp
except Exception as e:
	from torch.utils.cpp_extension import load
	clmvavg_cpp = load(name="clmvavg_cpp", sources=["utils/cpp/hplstm/clmvavg.cpp"], extra_cflags=extra_compile_args)
try:
	import ml2mvavgs_cuda
except Exception as e:
	if torch.cuda.is_available():
		from torch.utils.cpp_extension import load
		ml2mvavgs_cuda = load(name="ml2mvavgs_cuda", sources=["utils/cpp/hplstm/ml2mvavgs_cuda.cpp", "utils/cpp/hplstm/ml2mvavgs_cuda_kernel.cu"], extra_cflags=extra_compile_args, extra_cuda_cflags=extra_cuda_compile_args)
	else:
		ml2mvavgs_cuda = None

# memory friendly implementation of: _ = torch.cat((x.new_zeros(bsize, 1, nheads, adim), MvAvgFunc(x, ma_beta, False),), dim=1), (_.narrow(1, 0, seqlen), _.narrow(1, seqlen, 1))
class RS1MvAvgstatFunction(Function):

	@staticmethod
	def forward(ctx, x, beta):

		bsize, seqlen, nhead, isize = x.size()
		if seqlen > 1:
			_mem = x.new_empty(bsize, seqlen + 1, nhead, isize)
			_mem.select(1, 0).zero_()
			if x.is_cuda:
				ml2mvavgs_cuda.forward(x, _mem.narrow(1, 1, seqlen), beta, bsize, seqlen, nhead, isize)
			else:
				clmvavg_cpp.forward(_out.narrow(1, 1, seqlen).copy_(x), 1, beta)
			_out, _state = _mem.narrow(1, 0, seqlen), _mem.narrow(1, seqlen, 1)
		else:
			_out, _state = x.new_zeros(1, 1, 1, 1).expand(bsize, seqlen, nhead, isize), x.mul(1.0 - beta)
		ctx.beta = beta

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
						ml2mvavgs_cuda.backward(grad_x, grad_x, ctx.beta, bsize, seqlen, nhead, isize)
					else:
						clmvavg_cpp.backward(grad_x, 1, ctx.beta)
				else:
					_beta = ctx.beta
					grad_x.narrow(1, 0, 1).add_(grad_state, alpha=_beta).mul_(1.0 - _beta)

				return grad_x, None

		return None, None

RS1MvAvgstatFunc = RS1MvAvgstatFunction.apply
