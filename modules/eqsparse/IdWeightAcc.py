#encoding: utf-8

from torch.autograd import Function

from cnfg.ihyp import extra_compile_args, extra_cuda_compile_args

try:
	import idweightacc_cpp
except Exception as e:
	from torch.utils.cpp_extension import load
	idweightacc_cpp = load(name="idweightacc_cpp", sources=["modules/cpp/eqsparse/idweightacc.cpp"], extra_cflags=extra_compile_args + ["-fopenmp"])
try:
	import idweightacc_bias_cpp
except Exception as e:
	from torch.utils.cpp_extension import load
	idweightacc_bias_cpp = load(name="idweightacc_bias_cpp", sources=["modules/cpp/eqsparse/idweightacc_bias.cpp"], extra_cflags=extra_compile_args + ["-fopenmp"])
try:
	import idweightacc_cuda
except Exception as e:
	from torch.utils.cpp_extension import load
	idweightacc_cuda = load(name="idweightacc_cuda", sources=["modules/cpp/eqsparse/idweightacc_cuda.cpp", "modules/cpp/eqsparse/idweightacc_cuda_kernel.cu"], extra_cflags=extra_compile_args, extra_cuda_cflags=extra_cuda_compile_args)
try:
	import idweightacc_bias_cuda
except Exception as e:
	from torch.utils.cpp_extension import load
	idweightacc_bias_cuda = load(name="idweightacc_bias_cuda", sources=["modules/cpp/eqsparse/idweightacc_bias_cuda.cpp", "modules/cpp/eqsparse/idweightacc_bias_cuda_kernel.cu"], extra_cflags=extra_compile_args, extra_cuda_cflags=extra_cuda_compile_args)

class IdWeightAccFunction(Function):

	@staticmethod
	def forward(ctx, x, idx, weight, bias):

		bsize, isize = x.size()
		osize, ncon = weight.size()
		rs = x.new_empty(bsize, osize)
		rs = (idweightacc_cuda if x.is_cuda else idweightacc_cpp).forward_(x, idx, weight, rs, bsize, isize, osize, ncon) if bias is None else (idweightacc_bias_cuda if x.is_cuda else idweightacc_bias_cpp).forward_(x, idx, weight, bias, rs, bsize, isize, osize, ncon)
		ctx.save_for_backward(x, idx, weight)

		return rs

	@staticmethod
	def backward(ctx, grad_output):

		if ctx.needs_input_grad[3]:
			_ = grad_output.dim()
			grad_bias = grad_output if _== 1 else (grad_output if _ == 2 else grad_output.view(-1, grad_output.size(-1))).sum(0)
		else:
			grad_bias = None
		needs_grad_x, needs_grad_weight = ctx.needs_input_grad[0], ctx.needs_input_grad[2]
		if needs_grad_x or needs_grad_weight:
			x, idx, weight = ctx.saved_variables
			bsize, isize = x.size()
			osize, ncon = weight.size()
			grad_x = x.new_zeros(x.size())
			grad_weight = weight.new_zeros(weight.size())
			grad_x, grad_weight = (idweightacc_cuda if grad_output.is_cuda else idweightacc_cpp).backward_(x, idx, weight, grad_output, grad_x, grad_weight, bsize, isize, osize, ncon)#.contiguous()
			return grad_x if needs_grad_x else None, None, grad_weight if needs_grad_weight else None, grad_bias
		else:
			return None, None, None, grad_bias

IdWeightAccFunc = IdWeightAccFunction.apply
