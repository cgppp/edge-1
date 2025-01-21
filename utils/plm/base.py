#encoding: utf-8

import torch
from functools import wraps
from torch import nn

from modules.norm.base import RMSNorm
from utils.fmt.plm.base import fix_parameter_name
from utils.h5serial import h5File, h5ensure_tensor

from cnfg.ihyp import h5_fileargs

def copy_plm_parameter(src, plm_parameters, keys, func=None, func_args=None, func_kwargs=None, print_func=print):

	_tgt = None
	if isinstance(keys, str):
		_p_k = keys
		if keys in plm_parameters:
			_tgt = h5ensure_tensor(plm_parameters[keys])
	else:
		_p_k = str(keys)
		if all(_ in plm_parameters for _ in keys):
			_tgt = [h5ensure_tensor(plm_parameters[_]) for _ in keys]
	if _tgt is not None:
		if func is not None:
			_tgt = func(_tgt, *([] if func_args is None else func_args), **({} if func_kwargs is None else func_kwargs))
		_src = src
		_s_size, _t_size = _src.size(), _tgt.size()
		if len(_s_size) == len(_t_size):
			_mdl = []
			for _i, (_s, _t,) in enumerate(zip(_s_size, _t_size)):
				if _s > _t:
					_src = _src.narrow(_i, 0, _t)
					_mdl.append(_i)
				elif _s < _t:
					_tgt = _tgt.narrow(_i, 0, _s)
					_mdl.append(_i)
			_src.copy_(_tgt)
			if _mdl and (print_func is not None):
				print_func("size mismatch for %s at dimension(s) %s" % (_p_k, ",".join([str(_) for _ in _mdl]),))
				print_func(_s_size, _t_size)
		elif print_func is not None:
			print_func("dimension mismatch for %s" % _p_k)
			print_func(_s_size, _t_size)
	elif print_func is not None:
		print_func("%s does not exist" % _p_k)

def set_ln_ieps(netin, ieps):

	for net in netin.modules():
		if isinstance(net, (nn.LayerNorm, RMSNorm,)) and hasattr(net, "eps") and (net.eps != ieps):
			net.eps = ieps

	return netin

def load_plm_wrapper(fix_pname=True, torch_map_location="cpu"):

	def wrapper_builder(func):

		@wraps(func)
		def load_plm_wrapper_core(m, plmd, *args, **kwargs):

			if isinstance(plmd, str):
				if plmd.endswith(".h5"):
					with h5File(plmd, "r", **h5_fileargs) as _:
						if fix_pname:
							_ = fix_parameter_name(_)
						return func(m, _, *args, **kwargs)
				elif plmd.endswith(".bin"):
					_ = torch.load(plmd, map_location=torch_map_location)
					if fix_pname:
						_ = fix_parameter_name(_)
					return func(m, _, *args, **kwargs)

			return func(m, plmd, *args, **kwargs)

		return load_plm_wrapper_core

	return wrapper_builder
