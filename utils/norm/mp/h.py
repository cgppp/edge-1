#encoding: utf-8

import torch

from modules.norm.base import LayerNorm as fpLayerNorm, RMSNorm as fpRMSNorm
from modules.norm.mp.h import LayerNorm, RMSNorm
from utils.base import add_module, copy_module_parabuf, get_module_devtyp
from utils.torch.comp import fp16_default_tensor_type

def replace_fp_norm(modin, print_func=print, lpnorms=set([LayerNorm, RMSNorm]), **kwargs):

	_ns, _nts = [], set()
	for _name, _module in modin.named_modules():
		_mtype = type(_module)
		if _mtype == fpLayerNorm:
			_device, _dtype = get_module_devtyp(_module)
			_tmpm = LayerNorm(_module.normalized_shape, eps=_module.eps, elementwise_affine=_module.elementwise_affine, bias=_module.bias is not None, device=_device, dtype=_dtype)
			add_module(modin, _name, copy_module_parabuf(_module, _tmpm, sync_requires_grad=True))
		elif _mtype == fpRMSNorm:
			_device, _dtype = get_module_devtyp(_module)
			_tmpm = RMSNorm(_module.normalized_shape, eps=_module.eps, elementwise_affine=_module.elementwise_affine, device=_device, dtype=_dtype)
			add_module(modin, _name, copy_module_parabuf(_module, _tmpm, sync_requires_grad=True))
		elif isinstance(_module, (fpLayerNorm, fpRMSNorm,)) and (_mtype not in lpnorms) and (print_func is not None) and (_mtype not in _nts):
			_ns.append(_name)
			_nts.add(_mtype)
	if _ns:
		print_func("Not supported: %s" % (", ".join(_ns)))

	return modin

def convert(modin, device=None, dtype=fp16_default_tensor_type, non_blocking=True, **kwargs):

	replace_fp_norm(modin, **kwargs)
	modin.to(device=device, dtype=dtype, non_blocking=True)

	return modin
