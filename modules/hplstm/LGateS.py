#encoding: utf-8

from torch.autograd import Function

from cnfg.ihyp import extra_compile_args, extra_cuda_compile_args

try:
	import lgates_cpp
except Exception as e:
	from torch.utils.cpp_extension import load
	lgates_cpp = load(name="lgates_cpp", sources=["modules/cpp/hplstm/lgates.cpp"], extra_cflags=extra_compile_args + ["-fopenmp"])
try:
	import lgates_cuda
except Exception as e:
	from torch.utils.cpp_extension import load
	lgates_cuda = load(name="lgates_cuda", sources=["modules/cpp/hplstm/lgates_cuda.cpp", "modules/cpp/hplstm/lgates_cuda_kernel.cu"], extra_cflags=extra_compile_args, extra_cuda_cflags=extra_cuda_compile_args)

class LGateFunction(Function):

	@staticmethod
	def forward(ctx, fgate, igh, init_cell, inplace=False):

		_ = igh.size()
		cell = igh if inplace else igh.new_empty(_)
		cell = (lgates_cuda if igh.is_cuda else lgates_cpp).forward(fgate, igh, init_cell, cell, *_)
		ctx.save_for_backward(cell, fgate, init_cell)

		return cell

	@staticmethod
	def backward(ctx, grad_cell):

		needs_grad_fgate, needs_grad_igh, needs_grad_init_cell = ctx.needs_input_grad[0:3]
		if needs_grad_fgate or needs_grad_igh or needs_grad_init_cell:
			cell, fgate, init_cell = ctx.saved_variables
			grad_fgate = fgate.new_empty(fgate.size())
			grad_igh = cell.new_empty(cell.size())
			bsize, seqlen, nhead, isize = grad_cell.size()
			grad_init_cell = init_cell.new_empty(bsize, nhead, isize)
			grad_fgate, grad_igh, grad_init_cell = (lgates_cuda if grad_cell.is_cuda else lgates_cpp).backward(grad_cell, cell, fgate, init_cell, grad_fgate, grad_igh, grad_init_cell, bsize, seqlen, nhead, isize)
			return grad_fgate if needs_grad_fgate else None, grad_igh if needs_grad_igh else None, grad_init_cell.sum(0) if needs_grad_init_cell else None, None
		else:
			return None, None, None, None

LGateFunc = LGateFunction.apply
