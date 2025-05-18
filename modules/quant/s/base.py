#encoding: utf-8

import torch
from numbers import Integral
from torch import nn
from torch.nn import functional as nnFunc

from modules.norm.base import LayerNorm as LayerNormBase, RMSNorm as RMSNormBase, layer_norm, rms_norm
from utils.fmt.base import iter_to_str
from utils.fmt.parser import parse_none
from utils.fmt.quant import is_legal_dim, parse_dim
from utils.quant.s.proto.bsb import dequant, estimate_quant_hyp, quant
from utils.torch.comp import patch_nn_Module_setattr, torch_no_grad, torch_quant_dtype
from utils.torch.ext import is_floating_dtype

from cnfg.ihyp import *

patch_nn_Module_setattr()

class QPara(nn.Module):

	def __init__(self, data=None, quant_log_shift=0.0, dim=None, dtype=torch_quant_dtype, quant_io=False, **kwargs):

		super(QPara, self).__init__()

		self.dtype, self.quant_log_shift, self.dim, self.quant_io = None, quant_log_shift, parse_dim(dim, data=data), quant_io
		self.set_quant_dtype(dtype=dtype)
		if data is not None:
			self.quant(data=data, quant_log_shift=quant_log_shift, dim=dim)

	def forward(self, data=None, quant_log_shift=None, dim=None, dtype=None, quant_io=None, **kwargs):

		if data is None:

			return self.dequant(quant_log_shift=quant_log_shift, **kwargs)
		else:
			self.quant(data=data, quant_log_shift=quant_log_shift, dim=dim, dtype=dtype, **kwargs)

			return self.dequant(quant_log_shift=quant_log_shift, **kwargs) if parse_none(quant_io, self.quant_io) else data

	def size(self, *args, **kwargs):

		return self.data.size(*args, **kwargs)

	def quant(self, data=None, quant_log_shift=None, dim=None, dtype=None, **kwargs):

		self.set_quant_dtype(dtype=dtype)
		_quant_log_shift = parse_none(quant_log_shift, self.quant_log_shift)
		self.register_buffer("qhyp", estimate_quant_hyp(data, dim=parse_dim(dim, data=data) if is_legal_dim(dim) else self.dim, qmin=self.qmin, qmax=self.qmax, log_shift=_quant_log_shift), persistent=False)
		self.register_buffer("data", quant(data, self.qhyp, dtype=self.dtype, qmin=self.qmin, qmax=self.qmax, log_shift=_quant_log_shift), persistent=False)

	def dequant(self, data=None, quant_log_shift=None, **kwargs):

		return dequant(parse_none(data, self.data), self.qhyp, log_shift=parse_none(quant_log_shift, self.quant_log_shift), **kwargs)

	def set_quant_dtype(self, dtype=None, **kwargs):

		if (dtype is not None) and (dtype != self.dtype):
			self.dtype = dtype
			_is_floating_dtype = is_floating_dtype(dtype)
			_dtype_info = (torch.finfo if _is_floating_dtype else torch.iinfo)(dtype)
			self.qmin, self.qmax = float(_dtype_info.min), float(_dtype_info.max)

	def extra_repr(self):

		return ("%s, dtype={}, quant_log_shift={}, quant_dim={}, quant_io={}" % ", ".join(iter_to_str(self.size()))).format(self.dtype, self.quant_log_shift, self.dim, self.quant_io)

class Linear(nn.Linear):

	def __init__(self, in_features, out_features, bias=True, quant_log_shift=0.0, quant_dim=None, quant_bias=False, quant_io=False, **kwargs):

		super(Linear, self).__init__(in_features, out_features, bias=bias)

		self.quant_bias, self.quant_log_shift, self.quant_dim, self.quant_io = quant_bias, quant_log_shift, quant_dim, quant_io
		self.weight = QPara(data=self.weight, quant_log_shift=quant_log_shift, dim=quant_dim, quant_io=quant_io)
		if quant_bias and (self.bias is not None):
			self.bias = QPara(data=self.bias, quant_log_shift=quant_log_shift, dim=None, quant_io=quant_io)

	def forward(self, x, **kwargs):

		return nnFunc.linear(x, self.weight(), self.bias() if isinstance(self.bias, QPara) else self.bias)

	def from_std(self, m):

		self.weight(data=m.weight, quant_log_shift=self.quant_log_shift, dim=self.quant_dim, quant_io=False)
		if m.bias is None:
			if self.bias is not None:
				self.register_parameter("bias", None)
		else:
			if self.quant_bias:
				if self.bias is None:
					self.bias = QPara(data=m.bias, quant_log_shift=self.quant_log_shift, dim=None, quant_io=self.quant_io)
				else:
					self.bias(data=m.bias, quant_log_shift=self.quant_log_shift, dim=None, quant_io=False)
			else:
				self.bias = m.bias
		self.to(device=m.weight.device, dtype=m.weight.dtype, non_blocking=True)

	def to_std(self):

		out_features, in_features = self.weight.size()
		rs = nn.Linear(in_features, out_features, bias=self.bias is not None)
		_copy_bias = False
		if self.bias is not None:
			if self.quant_bias:
				_copy_bias = True
			else:
				rs.bias = self.bias
		with torch_no_grad():
			rs.weight.copy_(self.weight())
			if _copy_bias:
				rs.bias.copy_(self.bias())
		rs.to(device=self.weight.qhyp.device, dtype=self.weight.qhyp.dtype, non_blocking=True)

		return rs

class Embedding(nn.Embedding):

	def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, quant_log_shift=0.0, quant_dim=None, quant_io=False, **kwargs):

		super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight)

		self.quant_dim, self.quant_log_shift, self.quant_io = quant_dim, quant_log_shift, quant_io
		self.weight = QPara(data=self.weight, quant_log_shift=quant_log_shift, dim=quant_dim, quant_io=quant_io)
		self.index_dim = 1 if self.weight.qhyp.dim() > self.weight.data.dim() else 0

	def forward(self, x):

		_ = nnFunc.embedding(x, self.weight.data, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

		if self.weight.dim == 0:
			_flatten = x.dim() > 1
			_i = x.view(-1) if _flatten else x
			_qhyp = self.weight.qhyp.index_select(self.index_dim, _i)
			if _flatten:
				_qhyp = _qhyp.view(self.weight.qhyp.size(0), *x.size(), 1) if self.index_dim > 0 else _qhyp.view(*x.size(), 1)

			return dequant(_, _qhyp)
		else:

			return self.weight.dequant(data=_)

	def from_std(self, m):

		self.weight(data=m.weight, quant_log_shift=self.quant_log_shift, dim=self.quant_dim, quant_io=False)
		self.to(device=m.weight.device, dtype=m.weight.dtype, non_blocking=True)

	def to_std(self):

		num_embeddings, embedding_dim = self.weight.size()
		rs = nn.Embedding(num_embeddings, embedding_dim, padding_idx=self.padding_idx, max_norm=self.max_norm, norm_type=self.norm_type, scale_grad_by_freq=self.scale_grad_by_freq, sparse=self.sparse, _weight=nn.Parameter(self.weight()))

		rs.to(device=self.weight.qhyp.device, dtype=self.weight.qhyp.dtype, non_blocking=True)

		return rs

class LayerNorm(LayerNormBase):

	def __init__(self, normalized_shape, eps=ieps_ln_default, elementwise_affine=True, bias=True, device=None, dtype=None, quant_log_shift=0.0, quant_weight=False, quant_bias=False, quant_io=False, **kwargs):

		super(LayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias, device=device, dtype=dtype)

		self.quant_log_shift, self.quant_weight, self.quant_bias, self.quant_io = quant_log_shift, quant_weight, quant_bias, quant_io
		if quant_weight and (self.weight is not None):
			self.weight = QPara(data=self.weight, quant_log_shift=quant_log_shift, dim=None, quant_io=quant_io)
		if quant_bias and (self.bias is not None):
			self.bias = QPara(data=self.bias, quant_log_shift=quant_log_shift, dim=None, quant_io=quant_io)

	def forward(self, x, **kwargs):

		return layer_norm(x, self.normalized_shape, self.weight() if isinstance(self.weight, QPara) else self.weight, self.bias() if isinstance(self.bias, QPara) else self.bias, self.eps)

	def from_std(self, m):

		if m.weight is None:
			if self.weight is not None:
				self.register_parameter("weight", None)
		else:
			if self.quant_weight:
				if self.weight is None:
					self.weight = QPara(data=m.weight, quant_log_shift=self.quant_log_shift, dim=None, quant_io=self.quant_io)
				else:
					self.weight(data=m.weight, quant_log_shift=self.quant_log_shift, dim=None, quant_io=False)
			else:
				self.weight = m.weight
		if m.bias is None:
			if self.bias is not None:
				self.register_parameter("bias", None)
		else:
			if self.quant_bias:
				if self.bias is None:
					self.bias = QPara(data=m.bias, quant_log_shift=self.quant_log_shift, dim=None, quant_io=self.quant_io)
				else:
					self.bias(data=m.bias, quant_log_shift=self.quant_log_shift, dim=None, quant_io=False)
			else:
				self.bias = m.bias
		if m.weight is not None:
			self.to(device=m.weight.device, dtype=m.weight.dtype, non_blocking=True)

	def to_std(self):

		rs = LayerNormBase(self.weight.size(), eps=self.eps, elementwise_affine=self.weight is not None, bias=self.bias is not None, device=self.weight.qhyp.device, dtype=self.weight.qhyp.dtype)
		_copy_weight = False
		if self.weight is not None:
			if self.quant_weight:
				_copy_weight = True
			else:
				rs.weight = self.weight
		_copy_bias = False
		if self.bias is not None:
			if self.quant_bias:
				_copy_bias = True
			else:
				rs.bias = self.bias
		if _copy_weight or _copy_bias:
			with torch_no_grad():
				if _copy_weight:
					rs.weight.copy_(self.weight())
				if _copy_bias:
					rs.bias.copy_(self.bias())
		if self.weight is not None:
			rs.to(device=self.weight.qhyp.device, dtype=self.weight.qhyp.dtype, non_blocking=True)

		return rs

class RMSNorm(RMSNormBase):

	def __init__(self, normalized_shape, eps=ieps_ln_default, elementwise_affine=True, device=None, dtype=None, quant_log_shift=0.0, quant_weight=False, quant_io=False, **kwargs):

		super(RMSNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)

		self.quant_log_shift, self.quant_weight, self.quant_io = quant_log_shift, quant_weight, quant_io
		if quant_weight and (self.weight is not None):
			self.weight = QPara(data=self.weight, quant_log_shift=quant_log_shift, dim=None, quant_io=quant_io)

	def forward(self, x, **kwargs):

		return rms_norm(x, self.normalized_shape, self.weight() if isinstance(self.weight, QPara) else self.weight, self.eps)

	def from_std(self, m):

		if m.weight is None:
			if self.weight is not None:
				self.register_parameter("weight", None)
		else:
			if self.quant_weight:
				if self.weight is None:
					self.weight = QPara(data=m.weight, quant_log_shift=self.quant_log_shift, dim=None, quant_io=self.quant_io)
				else:
					self.weight(data=m.weight, quant_log_shift=self.quant_log_shift, dim=None, quant_io=False)
			else:
				self.weight = m.weight
		if m.weight is not None:
			self.to(device=m.weight.device, dtype=m.weight.dtype, non_blocking=True)

	def to_std(self):

		rs = RMSNormBase(self.weight.size(), eps=self.eps, elementwise_affine=self.weight is not None, device=self.weight.qhyp.device, dtype=self.weight.qhyp.dtype)
		if self.weight is not None:
			if self.quant_weight:
				with torch_no_grad():
					rs.weight.copy_(self.weight())
			else:
				rs.weight = self.weight
		if self.weight is not None:
			rs.to(device=self.weight.qhyp.device, dtype=self.weight.qhyp.dtype, non_blocking=True)

		return rs
