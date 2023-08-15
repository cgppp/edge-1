#encoding: utf-8

from torch.nn.modules.loss import _Loss

from utils.torch.comp import torch_any_wodim
from utils.torch.ext import cosim, pearson_corr

from cnfg.ihyp import ieps_ln_default

def token_loss_mask_reduction(loss, mask=None, reduction="mean"):

	_is_mean_reduction = reduction == "mean"
	if _is_mean_reduction:
		_num = loss.numel()
	if mask is not None:
		loss.masked_fill_(mask, 0.0)
		if _is_mean_reduction:
			_num -= mask.int().sum().item()
	if reduction != "none":
		loss = loss.sum()
	if _is_mean_reduction:
		loss = loss / float(_num)

	return loss

def cosim_loss(a, b, mask=None, dim=-1, eps=ieps_ln_default, reduction="mean"):

	return -token_loss_mask_reduction(cosim(a, b, dim=dim, keepdim=False, eps=eps), mask=mask, reduction=reduction)

class Cosim(_Loss):

	def __init__(self, dim=-1, eps=ieps_ln_default, reduction="mean", **kwargs):

		super(Cosim, self).__init__()
		self.dim, self.eps, self.reduction = dim, eps, reduction

	def forward(self, input, target, mask=None, **kwargs):

		return cosim_loss(input, target, mask=mask, dim=self.dim, eps=self.eps, reduction=self.reduction)

def pearson_loss(a, b, mask=None, dim=-1, eps=ieps_ln_default, reduction="mean"):

	return -token_loss_mask_reduction(pearson_corr(a, b, dim=dim, keepdim=False, eps=eps), mask=mask, reduction=reduction)

class PearsonCorr(Cosim):

	def forward(self, input, target, mask=None, **kwargs):

		return pearson_loss(input, target, mask=mask, dim=self.dim, eps=self.eps, reduction=self.reduction)

def scaledis_loss(a, b, mask=None, dim=-1, reduction="mean", sort=True, stable=False):

	if sort:
		_sb, _ib = b.sort(dim=dim, stable=stable)
		_sa = a.gather(dim, _ib)
	else:
		_sa, _sb = a, b
	_n = _sa.size(-1) - 1
	_ha, _ta = _sa.narrow(dim, 0, _n), _sa.narrow(dim, 1, _n)
	_hb, _tb = _sb.narrow(dim, 0, _n), _sb.narrow(dim, 1, _n)
	_zero_mask = _ta.eq(0.0) | _tb.eq(0.0)
	if torch_any_wodim(_zero_mask).item():
		loss = (_ha / _ta - _hb / _tb).masked_fill(_zero_mask, 0.0).abs().sum(dim) / (float(_n) - _zero_mask.to(a.dtype).sum(dim))
	else:
		loss = (_ha / _ta - _hb / _tb).abs().mean(dim)

	return token_loss_mask_reduction(loss, mask=mask, reduction=reduction)

class ScaleDis(_Loss):

	def __init__(self, dim=-1, reduction="mean", sort=True, stable=False, **kwargs):

		super(ScaleDis, self).__init__()
		self.dim, self.reduction, self.sort, self.stable = dim, reduction, sort, stable

	def forward(self, input, target, mask=None, **kwargs):

		return scaledis_loss(input, target, mask=mask, dim=self.dim, reduction=self.reduction, sort=self.sort, stable=self.stable)

def orderdis_loss(a, b, mask=None, dim=-1, reduction="mean", stable=False):

	_sa = a.gather(dim, b.argsort(dim=dim, descending=False, stable=stable))
	_n = _sa.size(-1) - 1

	return token_loss_mask_reduction((_sa.narrow(dim, 1, _n) - _sa.narrow(dim, 0, _n)).clamp_(min=0.0).sum(dim), mask=mask, reduction=reduction)

class OrderDis(_Loss):

	def __init__(self, dim=-1, reduction="mean", stable=False, **kwargs):

		super(OrderDis, self).__init__()
		self.dim, self.reduction, self.stable = dim, reduction, stable

	def forward(self, input, target, mask=None, **kwargs):

		return orderdis_loss(input, target, mask=mask, dim=self.dim, reduction=self.reduction, stable=self.stable)
