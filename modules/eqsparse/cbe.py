#encoding: utf-8

import torch

from modules.eqsparse.IdWeightAcc import IdWeightAccFunc
from modules.eqsparse.einsumbe import EqsLinear as EqsLinearBase

class EqsLinear(EqsLinearBase):

	def __init__(self, in_features, out_features, num_conn, bias=True, **kwargs):

		super(EqsLinear, self).__init__(in_features, out_features, num_conn, bias=bias, **kwargs)

		self.register_buffer("conn", self.conn.to(torch.int32, non_blocking=True), persistent=True)

	def forward(self, x, **kwargs):

		_x_size = x.size()

		return IdWeightAccFunc(x, self.conn, self.weight, self.bias) if len(_x_size) == 2 else IdWeightAccFunc(x.view(-1, _x_size[-1]), self.conn, self.weight, self.bias).view(*_x_size[:-1], self.weight.size(0))
