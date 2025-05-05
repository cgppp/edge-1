#encoding: utf-8

from torch import nn

from modules.lora import Embedding, Linear
from utils.base import add_module
from utils.func import always_true as name_cfunc_full

def lora_get_linear(modin, lora_features=None, lora_alpha=None, scaling=1.0, update_bias=True, **kwargs):

	out_features, in_features = modin.weight.size()
	_lora_m = Linear(in_features, out_features, bias=modin.bias is not None, lora_features=lora_features, lora_alpha=lora_alpha, scaling=scaling, update_bias=update_bias)
	_lora_m.from_std(modin)

	return _lora_m

def lora_get_embedding(modin, lora_features=None, lora_alpha=None, scaling=1.0, **kwargs):

	num_embeddings, embedding_dim = modin.weight.size()
	_lora_m = Embedding(num_embeddings, embedding_dim, padding_idx=modin.padding_idx, max_norm=modin.max_norm, norm_type=modin.norm_type, scale_grad_by_freq=modin.scale_grad_by_freq, sparse=modin.sparse, _weight=modin.weight, lora_features=lora_features, lora_alpha=lora_alpha, scaling=scaling)
	_lora_m.from_std(modin)

	return _lora_m

def std2lora(modin, lora_features=None, lora_alpha=None, scaling=1.0, update_bias=True, name_cfunc=name_cfunc_full):

	if isinstance(modin, nn.Linear):
		return lora_get_linear(modin, lora_features=lora_features, lora_alpha=lora_alpha, scaling=scaling, update_bias=update_bias), {".": modin}
	if isinstance(modin, nn.Embedding):
		return lora_get_embedding(modin, lora_features=lora_features, lora_alpha=lora_alpha, scaling=scaling), {".": modin}

	_l_d = {}
	for _name, _module in modin.named_modules():
		if name_cfunc(_name):
			if isinstance(_module, nn.Linear):
				_l_d[_name] = _module
				add_module(modin, _name, lora_get_linear(_module, lora_features=lora_features, lora_alpha=lora_alpha, scaling=scaling, update_bias=update_bias))
			elif isinstance(_module, nn.Embedding):
				_l_d[_name] = _module
				add_module(modin, _name, lora_get_embedding(_module, lora_features=lora_features, lora_alpha=lora_alpha, scaling=scaling))

	return modin, _l_d

def lora2std(modin):

	if isinstance(modin, (Linear, Embedding,)):
		return modin.to_std(), {".": modin}

	_lora_d = {}
	for _name, _module in modin.named_modules():
		if isinstance(_module, (Linear, Embedding,)):
			_lora_d[_name] = _module
			_std_m = _module.to_std()
			add_module(modin, _name, _std_m)

	return modin, _lora_d

def restore_md(modin, md):

	if "." in md:
		return md["."]

	for _name, _module in md.items():
		add_module(modin, _name, _module)

	return modin
