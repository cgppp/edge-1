#encoding: utf-8

from torch.autograd import Function

from cnfg.ihyp import extra_compile_args, extra_cuda_compile_args

try:
	import mlgatev_cpp
except Exception as e:
	from torch.utils.cpp_extension import load
	mlgatev_cpp = load(name="mlgatev_cpp", sources=["utils/cpp/hplstm/mlgatev.cpp"], extra_cflags=extra_compile_args)
try:
	import mlgates_cuda
except Exception as e:
	import torch
	if torch.cuda.is_available():
		from torch.utils.cpp_extension import load
		mlgates_cuda = load(name="mlgates_cuda", sources=["utils/cpp/hplstm/mlgates_cuda.cpp", "utils/cpp/hplstm/mlgates_cuda_kernel.cu"], extra_cflags=extra_compile_args, extra_cuda_cflags=extra_cuda_compile_args)
	else:
		mlgates_cuda = None

class LGateFunction(Function):

	# fgate: (bsize, seql, nheads, isize)
	# init_cell: (nheads, isize, nbank)
	# bfgate: (bsize, seql, nheads, nbank)
	# igh -> cell: (bsize, seql, nheads, isize, nbank)
	@staticmethod
	def forward(ctx, fgate, igh, init_cell, bfgate):

		cell = mlgates_cuda.forward(fgate, igh, init_cell, bfgate, *_) if igh.is_cuda else mlgatev_cpp.forward(fgate, igh, init_cell, bfgate, 1)
		ctx.save_for_backward(cell, fgate, init_cell, bfgate)

		return cell

	@staticmethod
	def backward(ctx, grad_cell):

		needs_grad_fgate, needs_grad_igh, needs_grad_init_cell, needs_grad_bfgate = ctx.needs_input_grad[0:4]
		if needs_grad_fgate or needs_grad_igh or needs_grad_init_cell or needs_grad_bfgate:
			cell, fgate, init_cell, bfgate = ctx.saved_variables
			if grad_cell.is_cuda:
				grad_fgate = fgate.new_empty(fgate.size())
				grad_igh = cell.new_empty(cell.size())
				bsize, seqlen, nhead, isize, nbank = grad_cell.size()
				grad_init_cell = init_cell.new_empty(bsize, nhead, isize, nbank)
				grad_bfgate = bfgate.new_empty(bfgate.size())
				grad_fgate, grad_igh, grad_init_cell, grad_bfgate = mlgates_cuda.backward(grad_cell, cell, fgate, init_cell, bfgate, grad_fgate, grad_igh, grad_init_cell, grad_bfgate, bsize, seqlen, nhead, isize, nbank)
			else:
				if needs_grad_fgate or needs_grad_bfgate:
					grad_fgate, grad_igh, grad_init_cell, grad_bfgate = mlgatev_cpp.backward(grad_cell, cell, fgate, init_cell, bfgate, 1)
				else:
					grad_igh, grad_init_cell = mlgate_cpp.backward_no_fgate(grad_cell, fgate, bfgate, 1)
					grad_fgate = grad_bfgate = None
			return grad_fgate if needs_grad_fgate else None, grad_igh if needs_grad_igh else None, grad_init_cell if needs_grad_init_cell else None, grad_bfgate if needs_grad_bfgate else None
		else:
			return None, None, None, None

LGateFunc = LGateFunction.apply
