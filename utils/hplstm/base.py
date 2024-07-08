#encoding: utf-8

import torch

def RS1CSum(x, dim):

	# following lines are memory friendly implementation of: torch.cat((x.new_zeros(bsize, 1, nheads, adim), x.narrow(1, 0, seql - 1),), dim=1).cumsum(dim=1)
	out = x.new_empty(x.size())
	out.select(dim, 0).zero_()
	_ = x.size(dim) - 1
	torch.cumsum(x.narrow(dim, 0, _), dim=dim, out=out.narrow(dim, 1, _))

	return out
