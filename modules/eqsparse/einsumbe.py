#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from utils.eqsparse import build_random_conn_inds
from utils.torch.comp import torch_no_grad

class EqsLinear(nn.Module):

	def __init__(self, in_features, out_features, num_conn, bias=True, **kwargs):

		super(EqsLinear, self).__init__()

		self.in_features, self.out_features, self.num_conn = in_features, out_features, num_conn
		self.weight = nn.Parameter(torch.Tensor(out_features, num_conn))
		self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
		self.register_buffer("conn", torch.tensor(build_random_conn_inds(in_features, out_features, num_conn), dtype=torch.long), persistent=True)
		self.reset_parameters()

	def forward(self, x, **kwargs):

		out = torch.einsum("...ab,ab->...a", x.index_select(-1, self.conn).view(*x.size()[:-1], self.out_features, self.num_conn), self.weight)
		if self.bias is not None:
			out.add_(self.bias)

		return out

	def extra_repr(self):

		return "in_features={}, out_features={}, num_connections={}, bias={}".format(self.in_features, self.out_features, self.num_conn, self.bias is not None)

	def reset_parameters(self):

		_ = self.num_conn
		with torch_no_grad():
			_bound = 1.0 / sqrt(_)
			self.weight.uniform_(-_bound, _bound)
			if self.bias is not None:
				self.bias.zero_()

	def fix_init(self):

		self.reset_parameters()
