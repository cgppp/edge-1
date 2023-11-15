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
		loss = loss.div_(float(_num))

	return loss

def cosim_loss(s, t, mask=None, dim=-1, reduction="mean", eps=ieps_ln_default):

	return -token_loss_mask_reduction(cosim(s, t, dim=dim, keepdim=False, eps=eps), mask=mask, reduction=reduction)

class Cosim(_Loss):

	def __init__(self, dim=-1, reduction="mean", eps=ieps_ln_default, **kwargs):

		super(Cosim, self).__init__()
		self.dim, self.reduction, self.eps = dim, reduction, eps

	def forward(self, input, target, mask=None, **kwargs):

		return cosim_loss(input, target, mask=mask, dim=self.dim, reduction=self.reduction, eps=self.eps)

def pearson_loss(s, t, mask=None, dim=-1, reduction="mean", eps=ieps_ln_default):

	return -token_loss_mask_reduction(pearson_corr(s, t, dim=dim, keepdim=False, eps=eps), mask=mask, reduction=reduction)

class PearsonCorr(Cosim):

	def forward(self, input, target, mask=None, **kwargs):

		return pearson_loss(input, target, mask=mask, dim=self.dim, reduction=self.reduction, eps=self.eps)

def scaledis_loss(s, t, mask=None, dim=-1, reduction="mean", sort=True, stable=False):

	if sort:
		_st, _it = t.sort(dim=dim, stable=stable)
		_ss = s.gather(dim, _it)
	else:
		_ss, _st = s, t
	_n = _ss.size(-1) - 1
	_hs, _ts = _ss.narrow(dim, 0, _n), _ss.narrow(dim, 1, _n)
	_ht, _tt = _st.narrow(dim, 0, _n), _st.narrow(dim, 1, _n)
	_zero_mask = _ts.eq(0.0) | _tt.eq(0.0)
	if torch_any_wodim(_zero_mask).item():
		loss = (_hs / _ts - _ht / _tt).masked_fill(_zero_mask, 0.0).abs().sum(dim) / (float(_n) - _zero_mask.to(s.dtype).sum(dim))
	else:
		loss = (_hs / _ts - _ht / _tt).abs().mean(dim)

	return token_loss_mask_reduction(loss, mask=mask, reduction=reduction)

class ScaleDis(_Loss):

	def __init__(self, dim=-1, reduction="mean", sort=True, stable=False, **kwargs):

		super(ScaleDis, self).__init__()
		self.dim, self.reduction, self.sort, self.stable = dim, reduction, sort, stable

	def forward(self, input, target, mask=None, **kwargs):

		return scaledis_loss(input, target, mask=mask, dim=self.dim, reduction=self.reduction, sort=self.sort, stable=self.stable)

def orderdis_loss(s, t, mask=None, dim=-1, reduction="mean", stable=False):

	_ss = s.gather(dim, t.argsort(dim=dim, descending=False, stable=stable))
	_n = _ss.size(-1) - 1

	return token_loss_mask_reduction((_ss.narrow(dim, 1, _n) - _ss.narrow(dim, 0, _n)).clamp_(min=0.0).mean(dim), mask=mask, reduction=reduction)

class OrderDis(_Loss):

	def __init__(self, dim=-1, reduction="mean", stable=False, **kwargs):

		super(OrderDis, self).__init__()
		self.dim, self.reduction, self.stable = dim, reduction, stable

	def forward(self, input, target, mask=None, **kwargs):

		return orderdis_loss(input, target, mask=mask, dim=self.dim, reduction=self.reduction, stable=self.stable)

def simorder_loss(s, t, mask=None, dim=-1, reduction="mean", alpha=0.5, eps=ieps_ln_default, stable=False, sim_func=pearson_corr):

	_sim = sim_func(s, t, dim=dim, keepdim=False, eps=eps)
	_ss = s.gather(dim, t.argsort(dim=dim, descending=False, stable=stable))
	_n = _ss.size(-1) - 1
	_order_loss = (_ss.narrow(dim, 1, _n) - _ss.narrow(dim, 0, _n)).clamp_(min=0.0).mean(dim)

	_is_mean_reduction = reduction == "mean"
	if _is_mean_reduction:
		_num = _sim.numel()
	if mask is not None:
		_sim.masked_fill_(mask, 0.0)
		_order_loss.masked_fill_(mask, 0.0)
		if _is_mean_reduction:
			_num -= mask.int().sum().item()
	if reduction != "none":
		_sim = _sim.sum()
		_order_loss = _order_loss.sum()
	loss = _order_loss.sub_(_sim) if alpha == 1.0 else (-_sim).add(_order_loss, alpha=alpha)
	if _is_mean_reduction:
		loss = loss.div_(float(_num))

	return loss

class SimOrder(_Loss):

	def __init__(self, dim=-1, reduction="mean", alpha=0.5, eps=ieps_ln_default, stable=False, sim_func=pearson_corr, **kwargs):

		super(SimOrder, self).__init__()
		self.dim, self.reduction, self.alpha, self.eps, self.stable, self.sim_func = dim, reduction, alpha, eps, stable, sim_func

	def forward(self, input, target, mask=None, **kwargs):

		return simorder_loss(input, target, mask=mask, dim=self.dim, reduction=self.reduction, alpha=self.alpha, eps=self.eps, stable=self.stable, sim_func=self.sim_func)
