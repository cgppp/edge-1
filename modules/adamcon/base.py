#encoding: utf-8

import torch
from math import sqrt
from numbers import Number
from torch import nn

from utils.torch.comp import torch_no_grad

from cnfg.ihyp import adam_betas_default, ieps_adam_default

class AdamCon(nn.Module):

	def __init__(self, base_lr=1.0, betas=adam_betas_default, eps=ieps_adam_default, lr_scale=1.0, **kwargs):

		super(AdamCon, self).__init__()

		self.base_lr, self.betas, self.eps, self.lr_scale = base_lr, betas, eps, lr_scale
		self.lr = base_lr * lr_scale
		self.reset()

	def forward(self, p, rg, **kwargs):

		beta1, beta2 = self.betas
		_step = self.step
		if (self.exp_avg is None) or (self.exp_avg_sq is None) or (_step == 1):
			self.exp_avg = rg * (1.0 - beta1)
			self.exp_avg_sq = rg.pow(2.0).mul_(1.0 - beta2)
		else:
			self.exp_avg = self.exp_avg.mul(beta1).add_(rg, alpha=1.0 - beta1)
			self.exp_avg_sq.mul_(beta2).addcmul_(rg, rg, value=1 - beta2)
		denom = self.exp_avg_sq.sqrt().add(self.eps)
		_step_size = self.get_lr() * sqrt(1.0 - beta2 ** _step) / (1.0 - beta1 ** _step)
		out = p.addcdiv(self.exp_avg, denom, value=_step_size) if isinstance(_step_size, Number) else p.addcmul(_step_size, self.exp_avg / denom)
		self.step = _step + 1

		return out

	def reset(self):

		self.step = 1
		self.exp_avg = self.exp_avg_sq = None

	def get_lr(self):

		return self.lr

class AvgAdamCon(AdamCon):

	def forward(self, p, rg, **kwargs):

		_step = self.step
		if (self.exp_avg is None) or (self.exp_avg_sq is None) or (_step == 1):
			_exp_avg = self.exp_avg = rg
			_exp_avg_sq = self.exp_avg_sq = rg.pow(2.0)
		else:
			_f_step = float(_step)
			self.exp_avg = self.exp_avg.add(rg)
			_exp_avg = self.exp_avg / _f_step
			self.exp_avg_sq.addcmul_(rg, rg)
			_exp_avg_sq = self.exp_avg_sq / _f_step
		denom = _exp_avg_sq.sqrt().add(self.eps)
		_step_size = self.get_lr()
		out = p.addcdiv(_exp_avg, denom, value=_step_size) if isinstance(_step_size, Number) else p.addcmul(_step_size, _exp_avg / denom)
		self.step = _step + 1

		return out

class DynLRAdamCon(AdamCon):

	def __init__(self, base_lr=1.0, betas=adam_betas_default, eps=ieps_adam_default, lr_scale=1.0, **kwargs):

		super(DynLRAdamCon, self).__init__(base_lr=base_lr, betas=betas, eps=eps, lr_scale=lr_scale * 2.0 / base_lr, **kwargs)

		self.base_lr = nn.Parameter(torch.zeros(1))

	def get_lr(self):

		return self.base_lr.sigmoid() * self.lr_scale

	def fix_init(self):

		with torch_no_grad():
			self.base_lr.data.zero_()
