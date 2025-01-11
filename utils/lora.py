#encoding: utf-8

from torch import nn

from modules.lora import Linear
from utils.base import add_module

def linear2lora(modin, lora_features=None, update_bias=True):

	_l_d = {}
	for _name, _module in modin.named_modules():
		if isinstance(_module, nn.Linear):
			_l_d[_name] = _module
			out_features, in_features = _module.weight.size()
			_lora_m = Linear(in_features, out_features, bias=_module.bias is not None, lora_features=lora_features, update_bias=update_bias)
			_lora_m.to(device=_module.weight.device, dtype=_module.weight.dtype, non_blocking=True)
			_lora_m.from_linear(_module)
			add_module(modin, _name, _lora_m)

	return modin, _l_d

def lora2linear(modin):

	_lora_d = {}
	for _name, _module in modin.named_modules():
		if isinstance(_module, Linear):
			_lora_d[_name] = _module
			_linear_m = _module.to_linear()
			add_module(modin, _name, _linear_m)

	return modin, _lora_d

def restore_md(modin, md):

	for _name, _module in md.items():
		add_module(modin, _name, _module)

	return modin
