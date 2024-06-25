#encoding: utf-8

from torch.autograd import Function
from torch.utils.cpp_extension import load

from cnfg.ihyp import extra_compile_args

try:
	import mvavg_cpp
except Exception as e:
	mvavg_cpp = load(name="mvavg_cpp", sources=["utils/cpp/mvavg.cpp"], extra_cflags=extra_compile_args)

class MvAvgFunction(Function):

	@staticmethod
	def forward(ctx, x, dim=None, beta=0.9, inplace=False):

		out = mvavg_cpp.forward(x, dim, beta, inplace)
		ctx.dim, ctx.beta = dim, beta

		return out

	@staticmethod
	def backward(ctx, grad_out):

		return mvavg_cpp.backward(grad_out, ctx.dim, ctx.beta), None, None, None

MvAvgFunc = MvAvgFunction.apply
