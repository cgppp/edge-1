#encoding: utf-8

import torch

from modules.norm.base import LayerNorm as LayerNormBase, RMSNorm as RMSNormBase, layer_norm, rms_norm

class LayerNorm(LayerNormBase):

	def forward(self, x, **kwargs):

		out = layer_norm(x.to(torch.float32, non_blocking=True), self.normalized_shape, None, None, self.eps).to(x.dtype, non_blocking=True)
		if self.weight is not None:
			if self.bias is None:
				out.mul_(self.weight)
			else:
				out = self.bias.addcmul(self.weight, out)
		elif self.bias is not None:
			out.add_(self.bias)

		return out

class RMSNorm(RMSNormBase):

	def forward(self, x, **kwargs):

		out = rms_norm(x.to(torch.float32, non_blocking=True), self.normalized_shape, None, self.eps).to(x.dtype, non_blocking=True)
		if self.weight is not None:
			_xn = _xn.mul_(self.weight)

		return out
