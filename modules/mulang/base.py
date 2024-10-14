#encoding: utf-8

import torch
from torch import nn
from torch.nn import functional as nnFunc

from utils.torch.comp import torch_no_grad

class MBLinear(nn.Linear):

	def __init__(self, in_features, out_features, nbias, bias=True, **kwargs):

		super(MBLinear, self).__init__(in_features, out_features, bias=False)

		if bias:
			self.bias = nn.Parameter(torch.zeros(nbias, out_features))

	def forward(self, x, taskid, **kwargs):

		out = nnFunc.linear(x, self.weight, None)
		if self.bias is not None:
			_bsize = [1 for i in range(x.dim())]
			_bsize[0] = x.size(0)
			_bsize[-1] = self.out_features
			out.add_(self.bias.index_select(0, taskid).view(_bsize))

		return out

	def fix_init(self):

		if self.bias is not None:
			with torch_no_grad():
				self.bias.zero_()

class NWMBLinear(MBLinear):

	def forward(self, x, taskid, **kwargs):

		out = nnFunc.linear(x, self.weight.narrow(0, 0, self.out_features), None)
		if self.bias is not None:
			_bsize = [1 for i in range(x.dim())]
			_bsize[0] = x.size(0)
			_bsize[-1] = self.out_features
			out.add_(self.bias.index_select(0, taskid).view(_bsize))

		return out
