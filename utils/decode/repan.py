#encoding: utf-8

import torch

from utils.fmt.parser import parse_none
from utils.torch.comp import torch_any_wodim

def penalty(x, inds, penalty=1.0, dim=-1, inplace=False):

	_x = x
	if penalty != 1.0:
		_ = _x.gather(dim, inds)
		_m = _.lt(0.0)
		if torch_any_wodim(_m):
			_[_m] *= penalty
		_m = ~_m
		if torch_any_wodim(_m):
			_[_m] /= penalty
		if inplace:
			_x.scatter_(dim, inds, _)
		else:
			_x = _x.scatter(dim, inds, _)

	return _x

class Penalty:

	def __init__(self, penalty=1.0, dim=-1, inplace=False, ind_unsqueeze_dim=1, **kwargs):

		self.penalty, self.dim, self.inplace, self.ind_unsqueeze_dim = penalty, dim, inplace, ind_unsqueeze_dim
		self.clear()

	def __call__(self, x, hist=None, **kwargs):

		_x, _hist = x, parse_none(hist, self.hist)
		if _hist is not None:
			_x = penalty(_x, _hist, penalty=self.penalty, dim=self.dim, inplace=self.inplace)

		return _x

	def record(self, inds=None):

		if (self.penalty != 1.0) and (inds is not None):
			_ = inds if self.ind_unsqueeze_dim is None else inds.unsqueeze(self.ind_unsqueeze_dim)
			self.hist = _ if self.hist is None else torch.cat((self.hist, _,), dim=self.dim)

		return self

	def clear(self):

		self.hist = None

		return self

	def act(self):

		return self.penalty != 1.0
