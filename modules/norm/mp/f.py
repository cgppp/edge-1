#encoding: utf-8

from modules.norm.base import LayerNorm as LayerNormBase, RMSNorm as RMSNormBase, layer_norm, rms_norm

class LayerNorm(LayerNormBase):

	def forward(self, x, **kwargs):

		return layer_norm(x.float(), self.normalized_shape, self.weight, self.bias, self.eps).to(x.dtype, non_blocking=True)

class RMSNorm(RMSNormBase):

	def forward(self, x, **kwargs):

		return rms_norm(x.float(), self.normalized_shape, self.weight, self.eps).to(x.dtype, non_blocking=True)
