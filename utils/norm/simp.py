#encoding: utf-8

from modules.norm.exp import SimpNorm
from utils.base import add_module, copy_module_parabuf

def simplify_ln(modin):

	for _name, _module in modin.named_modules():
		if isinstance(_module, LayerNorm):
			_tmpm = SimpNorm(_module.normalized_shape if _module.weight is None else tuple(_module.weight.size()), eps=_module.eps, elementwise_affine=_module.elementwise_affine)
			add_module(modin, _name, copy_module_parabuf(_module, _tmpm, sync_requires_grad=True))

	return modin
