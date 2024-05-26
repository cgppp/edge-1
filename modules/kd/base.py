#encoding: utf-8

import torch
from torch import nn
from torch.autograd import Function

from utils.torch.comp import torch_no_grad
from utils.torch.ext import cosim as sim_func, multinomial

class Argmax(nn.Module):

	def __init__(self, dim=-1, keepdim=False, **kwargs):

		super(Argmax, self).__init__()
		self.dim, self.keepdim = dim, keepdim

	def forward(self, x, **kwargs):

		return x.argmax(self.dim, keepdim=self.keepdim)

# nn.Module version of utils.sampler.SampleMax
class SampleMax(Argmax):

	def forward(self, x, **kwargs):

		out = multinomial(x, 1, replacement=True, dim=self.dim)

		return out if self.keepdim else out.squeeze(self.dim)

class DynT(nn.Module):

	def __init__(self, init_value=1.0, min_value=None, **kwargs):

		super(DynT, self).__init__()
		_init_value = abs(init_value)
		self.min_value = (_init_value / 64.0) if min_value is None else min_value
		self.init_value = (_init_value - self.min_value) if _init_value >= self.min_value else 0.0
		self.T = nn.Parameter(torch.as_tensor([self.init_value]))

	def forward(self, *args, **kwargs):

		return self.T.abs() + self.min_value

	def fix_init(self):

		with torch_no_grad():
			self.T.data.fill_(self.init_value)

class TSoftMax(nn.Module):

	def __init__(self, T=None, inplace=True, dim=-1, **kwargs):

		super(TSoftMax, self).__init__()

		self.T, self.inplace, self.dim = 1.0 / float(T), inplace, dim

	def forward(self, x, **kwargs):

		if self.T != 1.0:
			out = x.mul_(self.T) if self.inplace else x.mul(self.T)
		else:
			out = x

		return out.softmax(self.dim)

class TLogSoftMax(nn.Module):

	def __init__(self, T=None, dim=-1, **kwargs):

		super(TLogSoftMax, self).__init__()

		self.T = 1.0 / float(T)
		self.lsm = nn.LogSoftmax(dim)# for memory inefficient: if dim < 0 else (dim + 1)

	def forward(self, x, **kwargs):

		out = self.lsm(x)
		if self.training:
			if self.T != 1.0:
				# less memory efficient: self.lsm(torch.cat((x, x.mul(self.T),), dim=0)).unbind(0)
				return out, self.lsm(x.mul(self.T))
			else:
				return out, out
		else:
			return out

class GradientAdapterFunction(Function):

	@staticmethod
	def forward(ctx, inputs, min_sim=0.0):

		ctx.min_sim = min_sim

		return inputs, inputs

	@staticmethod
	def backward(ctx, grad_main, grad_sub):

		_grad_input = None
		if (grad_main is not None) and ctx.needs_input_grad[0]:
			_min_sim = ctx.min_sim
			if _min_sim == 0.0:
				_grad_input = grad_main.addcmul(sim_func(grad_main, grad_sub, dim=-1, keepdim=True).clamp_(min=0.0), grad_sub)
			else:
				_ = sim_func(grad_main, grad_sub, dim=-1, keepdim=True)
				_grad_input = grad_main.addcmul(_.masked_fill_(_.lt(_min_sim), 0.0), grad_sub)

		return _grad_input, None

GradientAdapterFunc = GradientAdapterFunction.apply

class GradientAdapter(nn.Module):

	def __init__(self, min_sim=0.0, **kwargs):

		super(GradientAdapter, self).__init__()

		self.min_sim = min_sim

	def forward(self, x, **kwargs):

		return GradientAdapterFunc(x, self.min_sim)

class MGradientAdapterFunction(Function):

	@staticmethod
	def forward(ctx, inputs, min_sim=0.0, k=2):

		ctx.min_sim = min_sim

		return tuple(inputs for _ in range(k))

	@staticmethod
	def backward(ctx, grad_main, *grad_sub):

		_grad_input = None
		if (grad_main is not None) and ctx.needs_input_grad[0]:
			_min_sim = ctx.min_sim
			if _min_sim == 0.0:
				for _grad_subu in grad_sub:
					_grad_input = (grad_main if _grad_input is None else _grad_input).addcmul(sim_func(grad_main, _grad_subu, dim=-1, keepdim=True).clamp_(min=0.0), _grad_subu)
				else:
					_grad_input = grad_main
			else:
				for _grad_subu in grad_sub:
					_ = sim_func(grad_main, _grad_subu, dim=-1, keepdim=True)
					_grad_input = (grad_main if _grad_input is None else _grad_input).addcmul(_.masked_fill_(_.lt(_min_sim), 0.0), _grad_subu)
				else:
					_grad_input = grad_main

		return _grad_input, None, None

MGradientAdapterFunc = MGradientAdapterFunction.apply

class MGradientAdapter(nn.Module):

	def __init__(self, min_sim=0.0, k=2, **kwargs):

		super(MGradientAdapter, self).__init__()

		self.min_sim, self.k = min_sim, k

	def forward(self, x, **kwargs):

		return MGradientAdapterFunc(x, self.min_sim, self.k)
