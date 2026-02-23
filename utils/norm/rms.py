#encoding: utf-8

from modules.norm.base import LayerNorm, RMSNorm
from utils.base import add_module, copy_module_parabuf, get_module_devtyp

def ln2rms(modin, **kwargs):

	for _name, _module in modin.named_modules():
		if type(_module) == LayerNorm:
			_device, _dtype = get_module_devtyp(_module)
			_tmpm = RMSNorm(_module.normalized_shape, eps=_module.eps, elementwise_affine=_module.elementwise_affine, device=_device, dtype=_dtype)
			add_module(modin, _name, copy_module_parabuf(_module, _tmpm, sync_requires_grad=True, print_func=None))

	return modin

def rms2ln(modin, bias=True, **kwargs):

	for _name, _module in modin.named_modules():
		if type(_module) == RMSNorm:
			_device, _dtype = get_module_devtyp(_module)
			_tmpm = LayerNorm(_module.normalized_shape, eps=_module.eps, elementwise_affine=_module.elementwise_affine, bias=bias, device=_device, dtype=_dtype)
			add_module(modin, _name, copy_module_parabuf(_module, _tmpm, sync_requires_grad=True))

	return modin
