#encoding: utf-8

from torch import nn

from modules.norm.base import LayerNorm as StdLayerNorm, RMSNorm as StdRMSNorm
from modules.quant.s.base import Embedding, LayerNorm, Linear, QPara, RMSNorm
from utils.base import add_module, get_parameter_tying
from utils.func import always_true as name_cfunc_full
from utils.module.patcher import patch_std, to_std

def quant_get_linear(modin, quant_dim=None, quant_bias=False, quant_io=False, **kwargs):

	if isinstance(modin, Linear):

		return modin

	out_features, in_features = modin.weight.size()
	rsm = Linear(in_features, out_features, bias=modin.bias is not None, quant_dim=quant_dim, quant_bias=quant_bias, quant_io=quant_io)
	rsm.from_std(modin)

	return rsm

def quant_get_embedding(modin, quant_dim=None, quant_io=False, **kwargs):

	if isinstance(modin, Embedding):

		return modin

	num_embeddings, embedding_dim = modin.weight.size()
	rsm = Embedding(num_embeddings, embedding_dim, padding_idx=modin.padding_idx, max_norm=modin.max_norm, norm_type=modin.norm_type, scale_grad_by_freq=modin.scale_grad_by_freq, sparse=modin.sparse, _weight=modin.weight, quant_dim=quant_dim, quant_io=quant_io)
	rsm.from_std(modin)

	return rsm

def quant_get_layernorm(modin, quant_dim=None, quant_weight=False, quant_bias=False, quant_io=False, **kwargs):

	if isinstance(modin, LayerNorm):

		return modin

	rsm = LayerNorm(modin.normalized_shape if modin.weight is None else modin.weight.size(), eps=modin.eps, elementwise_affine=modin.weight is not None, bias=modin.bias is not None, quant_dim=quant_dim, quant_weight=quant_weight, quant_bias=quant_bias, quant_io=quant_io)
	rsm.from_std(modin)

	return rsm

def quant_get_rmsnorm(modin, quant_dim=None, quant_weight=False, quant_io=False, **kwargs):

	if isinstance(modin, RMSNorm):

		return modin

	rsm = RMSNorm(modin.normalized_shape if modin.weight is None else modin.weight.size(), eps=modin.eps, elementwise_affine=modin.weight is not None, quant_dim=quant_dim, quant_weight=quant_weight, quant_io=quant_io)
	rsm.from_std(modin)

	return rsm

type_func = {nn.Linear: quant_get_linear, nn.Embedding: quant_get_embedding, StdLayerNorm: quant_get_layernorm, StdRMSNorm: quant_get_rmsnorm}
tgt_types = (Linear, Embedding, LayerNorm, RMSNorm,)

def quant(modin, quant_linear=True, quant_embedding=True, quant_normer=False, quant_dim=None, quant_weight=False, quant_bias=False, quant_io=False, name_cfunc=name_cfunc_full, keep_tying=True, type_func=type_func, **kwargs):

	_ = {}
	if quant_linear:
		_[nn.Linear] = type_func[nn.Linear]
	if quant_embedding:
		_[nn.Embedding] = type_func[nn.Embedding]
	if quant_normer:
		_[StdLayerNorm] = type_func[StdLayerNorm]
		_[RMSNorm] = type_func[RMSNorm]

	_tl = get_parameter_tying(modin) if keep_tying and _ else []
	rsm, md = patch_std(modin, quant_dim=quant_dim, quant_weight=quant_weight, quant_bias=quant_bias, quant_io=quant_io, type_pfunc=_, name_cfunc=name_cfunc, **kwargs) if _ else (modin, {},)
	if _tl:
		_mpd = {_n: _m for _n, _m in rsm.named_modules(remove_duplicate=False) if isinstance(_m, QPara)}
		if _mpd:
			for _nl in _tl:
				_el = [_ for _ in _nl if _ in _mpd]
				if len(_el) > 1:
					_m = _mpd[_el[0]]
					for _n in _el[1:]:
						if _m != _mpd[_n]:
							add_module(rsm, _n, _m)

	return rsm, md

def dequant(modin, types=tgt_types):

	return to_std(modin, types=types)
