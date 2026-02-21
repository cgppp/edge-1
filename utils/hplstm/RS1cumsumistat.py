#encoding: utf-8

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

class RS1cumsumistatFunction(Function):

	@staticmethod
	def forward(ctx, x, state):

		bsize, seqlen, nhead, isize = x.size()
		if seqlen > 1:
			_mem = x.new_empty(bsize, seqlen + 1, nhead, isize)
			_mem.narrow(1, 0, 1).copy_(state)
			_mem.narrow(1, 1, seqlen).copy_(x)
			_mem.cumsum_(dim=1)
			_out, _state = _mem.narrow(1, 0, seqlen), _mem.narrow(1, seqlen, 1)
		else:
			_out, _state = state, x + state

		return _out, _state

	@staticmethod
	def backward(ctx, grad_out, grad_state):

		grad_x = grad_s = None
		needs_grad_x, needs_grad_s = ctx.needs_input_grad[0:2]
		if needs_grad_x:
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
					grad_x.narrow(1, 0, 1).add_(grad_state)
		if needs_grad_s:
			grad_s = (grad_x.narrow(1, 0, 1) + grad_out.narrow(1, 0, 1)) if needs_grad_x else grad_out.sum(dim=1, keepdim=True).add_(grad_state)

		return grad_x, grad_s

RS1cumsumistatFunc = RS1cumsumistatFunction.apply
