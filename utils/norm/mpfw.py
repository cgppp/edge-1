#encoding: utf-8

from modules.norm.base import LayerNorm as fpLayerNorm, RMSNorm as fpRMSNorm
from modules.norm.mpfw import LayerNorm, RMSNorm
from utils.base import add_module, copy_module_parabuf

def replace_fp_norm(modin):

	for _name, _module in modin.named_modules():
		if isinstance(_module, fpLayerNorm):
			_tmpm = LayerNorm(_module.normalized_shape, eps=_module.eps, elementwise_affine=_module.elementwise_affine, bias=_module.bias is not None)
			add_module(modin, _name, copy_module_parabuf(_module, _tmpm))
		elif isinstance(_module, fpRMSNorm):
			_tmpm = RMSNorm(_module.normalized_shape, eps=_module.eps, elementwise_affine=_module.elementwise_affine)
			add_module(modin, _name, copy_module_parabuf(_module, _tmpm))

	return modin

def norm_para_fp32(modin):

	for _m in modin.modules():
		if isinstance(_m, (LayerNorm, RMSNorm,)):
			_m.float()

	return modin
