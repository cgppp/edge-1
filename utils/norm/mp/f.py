#encoding: utf-8

import torch
from functools import wraps

from modules.norm.base import LayerNorm as fpLayerNorm, RMSNorm as fpRMSNorm
from modules.norm.mp.f import LayerNorm, RMSNorm
from utils.base import add_module, copy_module_parabuf, get_module_devtyp
from utils.torch.comp import fp16_default_tensor_type, low_precision_floats

default_fp32_forward_wrapper = "xak"

def fp32_forward_wrapper_x(func):

	@wraps(func)
	def wrap_core_x(m, x, *args, **kwargs):

		return func(m, x.to(torch.float32, non_blocking=True), *args, **kwargs).to(x.dtype, non_blocking=True)

	return wrap_core_x

def fp32_forward_wrapper_xak(func):

	@wraps(func)
	def wrap_core_xak(m, x, *args, **kwargs):

		return func(m, x.to(torch.float32, non_blocking=True), *tuple((_.to(torch.float32, non_blocking=True) if isinstance(_, torch.Tensor) and (_.dtype in low_precision_floats) else _) for _ in args), **{_k: (_v.to(torch.float32, non_blocking=True) if isinstance(_v, torch.Tensor) and (_v.dtype in low_precision_floats) else _v) for _k, _v in kwargs.items()}).to(x.dtype, non_blocking=True)

	return wrap_core_xak

def fp32_forward_wrapper_general(func):

	@wraps(func)
	def wrap_core_general(*args, **kwargs):

		_dtype = None
		_args = []
		for _ in args:
			if isinstance(_, torch.Tensor):
				_xdtype = _.dtype
				if _xdtype in low_precision_floats:
					if _dtype is None:
						_dtype = _xdtype
					_args.append(_.to(torch.float32, non_blocking=True))
				else:
					_args.append(_)
			else:
				_args.append(_)
		_kwargs = {}
		for _k, _v in kwargs.items():
			if isinstance(_v, torch.Tensor):
				_vdtype = _v.dtype
				if _vdtype in low_precision_floats:
					if _dtype is None:
						_dtype = _vdtype
					_kwargs[_k] = _v.to(torch.float32, non_blocking=True)
				else:
					_kwargs[_k] = _v
			else:
				_kwargs[_k] = _v

		return func(*args, **kwargs) if _dtype is None else func(*_args, **_kwargs).to(_dtype, non_blocking=True)

	return wrap_core_general

fp32_forward_wrapper_dict = {"x": fp32_forward_wrapper_x, "xak": fp32_forward_wrapper_xak, "gen": fp32_forward_wrapper_general}
fp32_forward_wrapper = fp32_forward_wrapper_dict.get(default_fp32_forward_wrapper, fp32_forward_wrapper_general)

def replace_fp_norm(modin, print_func=print, lpnorms=set([LayerNorm, RMSNorm]), try_wrapper=True, **kwargs):

	_ws, _ns, _wts, _nts = [], [], set(), set()
	for _name, _module in modin.named_modules():
		_mtype = type(_module)
		if _mtype == fpLayerNorm:
			_device = get_module_devtyp(_module)[0]
			_tmpm = LayerNorm(_module.normalized_shape, eps=_module.eps, elementwise_affine=_module.elementwise_affine, bias=_module.bias is not None, device=_device, dtype=torch.float32)
			add_module(modin, _name, copy_module_parabuf(_module, _tmpm))
		elif _mtype == fpRMSNorm:
			_device = get_module_devtyp(_module)[0]
			_tmpm = RMSNorm(_module.normalized_shape, eps=_module.eps, elementwise_affine=_module.elementwise_affine, device=_device, dtype=torch.float32)
			add_module(modin, _name, copy_module_parabuf(_module, _tmpm))
		elif isinstance(_module, (fpLayerNorm, fpRMSNorm,)) and (_mtype not in lpnorms):
			if try_wrapper and hasattr(_module, "forward"):
				_module.to(torch.float32, non_blocking=True)
				if _mtype not in _wts:
					_mtype.forward = fp32_forward_wrapper(_mtype.forward)
					_ws.append(_name)
					_wts.add(_mtype)
			elif _mtype not in _nts:
				_ns.append(_name)
				_nts.add(_mtype)
	if print_func is not None:
		if _ws:
			print_func("Try to support with wrapper: %s" % (", ".join(_ws)))
		if _ns:
			print_func("Not supported: %s" % (", ".join(_ns)))

	return modin

def norm_para_fp32(modin):

	for _m in modin.modules():
		if isinstance(_m, (fpLayerNorm, fpRMSNorm,)):
			_m.to(torch.float32, non_blocking=True)

	return modin

def convert(modin, device=None, dtype=fp16_default_tensor_type, non_blocking=True, try_wrapper=True, **kwargs):

	modin.to(device=device, dtype=dtype, non_blocking=True)
	replace_fp_norm(modin, try_wrapper=try_wrapper, **kwargs)

	return modin
