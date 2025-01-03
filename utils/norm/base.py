#encoding: utf-8

from torch.nn import LayerNorm

def disable_ln_bias(modin):

	for _m in modin.modules():
		if isinstance(_m, LayerNorm):
			_m.register_parameter("bias", None)

	return modin
