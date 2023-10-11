#encoding: utf-8

from math import sqrt
from torch.nn import Embedding, LayerNorm, Linear
from torch.nn.init import _calculate_fan_in_and_fan_out

from utils.torch.comp import torch_no_grad

from cnfg.hyp import lipschitz_scale

def xavier_uniform_(tensor, gain=1.0):

	_scale = sqrt(6.0)
	if gain is not None and gain > 0.0 and gain != 1.0:
		_scale *= gain

	if tensor.requires_grad and (tensor.dim() > 1):
		with torch_no_grad():
			_fin, _fo = _calculate_fan_in_and_fan_out(tensor)
			_bound = _scale / sqrt(float(_fin + _fo))
			tensor.uniform_(-_bound, _bound)

	return tensor

def kaiming_uniform_(tensor, gain=1.0):

	_scale = sqrt(3.0)
	if gain is not None and gain > 0.0 and gain != 1.0:
		_scale *= gain

	if tensor.requires_grad and (tensor.dim() > 1):
		with torch_no_grad():
			_fin, _ = _calculate_fan_in_and_fan_out(tensor)
			_bound = _scale / sqrt(float(_fin))
			tensor.uniform_(-_bound, _bound)

	return tensor

def init_model_params_glorot(modin, gain=1.0, **kwargs):

	_scale = sqrt(6.0)
	if gain is not None and gain > 0.0 and gain != 1.0:
		_scale *= gain
	with torch_no_grad():
		for p in modin.parameters():
			if p.requires_grad and (p.dim() > 1):
				_fin, _fo = _calculate_fan_in_and_fan_out(p)
				_bound = _scale / sqrt(float(_fin + _fo))
				p.uniform_(-_bound, _bound)

	return modin

def init_model_params_kaiming(modin, gain=1.0, **kwargs):

	_scale = sqrt(3.0)
	if gain is not None and gain > 0.0 and gain != 1.0:
		_scale *= gain
	with torch_no_grad():
		for p in modin.parameters():
			if p.requires_grad and (p.dim() > 1):
				_fin, _ = _calculate_fan_in_and_fan_out(p)
				_bound = _scale / sqrt(float(_fin))
				p.uniform_(-_bound, _bound)

	return modin

def init_model_params_lipschitz(modin, gain_glorot=sqrt(1.0/3.0), gain_kaiming=sqrt(1.0/3.0), lipschitz_scale=lipschitz_scale, **kwargs):

	_gain_glorot, _gain_kaiming = gain_glorot, gain_kaiming
	if lipschitz_scale != 1.0:
		_gain_glorot *= lipschitz_scale
		_gain_kaiming *= lipschitz_scale

	_tmpm = init_model_params_kaiming(modin, gain=_gain_kaiming)

	with torch_no_grad():
		for _m in _tmpm.modules():
			if isinstance(_m, Embedding):
				init_model_params_glorot(_m, gain=_gain_glorot)
				if _m.padding_idx is not None:
					_m.weight[_m.padding_idx].zero_()
			elif isinstance(_m, Linear):
				if _m.bias is not None:
					_m.bias.zero_()
			elif isinstance(_m, LayerNorm):
				if _m.weight is not None:
					_m.weight.fill_(1.0)
				if _m.bias is not None:
					_m.bias.zero_()

	return _tmpm

def init_model_params(modin, lipschitz_scale=lipschitz_scale, **kwargs):

	return init_model_params_glorot(modin, gain=1.0, **kwargs) if lipschitz_scale is None else init_model_params_lipschitz(modin, lipschitz_scale=lipschitz_scale, **kwargs)
