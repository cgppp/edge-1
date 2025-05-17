#encoding: utf-8

from torch import nn

from modules.lora.base import Embedding, Linear
from utils.base import add_parameter, get_parameter_tying
from utils.func import always_true as name_cfunc_full
from utils.module.patcher import patch_std, to_std

def lora_get_linear(modin, lora_features=None, lora_alpha=None, scaling=1.0, update_bias=True, **kwargs):

	if isinstance(modin, Linear):

		return modin

	out_features, in_features = modin.weight.size()
	rsm = Linear(in_features, out_features, bias=modin.bias is not None, lora_features=lora_features, lora_alpha=lora_alpha, scaling=scaling, update_bias=update_bias)
	rsm.from_std(modin)

	return rsm

def lora_get_embedding(modin, lora_features=None, lora_alpha=None, scaling=1.0, **kwargs):

	if isinstance(modin, Embedding):

		return modin

	num_embeddings, embedding_dim = modin.weight.size()
	rsm = Embedding(num_embeddings, embedding_dim, padding_idx=modin.padding_idx, max_norm=modin.max_norm, norm_type=modin.norm_type, scale_grad_by_freq=modin.scale_grad_by_freq, sparse=modin.sparse, _weight=modin.weight, lora_features=lora_features, lora_alpha=lora_alpha, scaling=scaling)
	rsm.from_std(modin)

	return rsm

type_func = {nn.Linear: lora_get_linear, nn.Embedding: lora_get_embedding}
tgt_types = (Linear, Embedding,)

def std2lora(modin, lora_features=None, lora_alpha=None, scaling=1.0, update_bias=True, name_cfunc=name_cfunc_full, keep_lora_weight_tying=True, type_func=type_func, **kwargs):

	_tl = get_parameter_tying(modin) if keep_lora_weight_tying else []
	rsm, md = patch_std(modin, lora_features=lora_features, lora_alpha=lora_alpha, scaling=scaling, update_bias=update_bias, type_pfunc=type_func, name_cfunc=name_cfunc, **kwargs)
	if _tl:
		_mpd = dict(rsm.named_parameters(remove_duplicate=False))
		for _nl in _tl:
			_el = [_ for _ in _nl if _ in _mpd]
			if len(_el) > 1:
				_p = _mpd[_el[0]]
				for _n in _el[1:]:
					if not _p.is_set_to(_mpd[_n]):
						add_parameter(rsm, _n, _p)
				_el = [_ for _ in _nl if (("%slora_wa" % _[:-6]) in _mpd) and (("%slora_wb" % _[:-6]) in _mpd)]
				if len(_el) > 1:
					_pa, _pb = _mpd["%slora_wa" % _el[0][:-6]], _mpd["%slora_wb" % _el[0][:-6]]
					for _n in _el[1:]:
						_na, _nb = "%slora_wa" % _n[:-6], "%slora_wb" % _n[:-6]
						if not _pa.is_set_to(_mpd[_na]):
							add_parameter(rsm, _na, _pa)
						if not _pb.is_set_to(_mpd[_nb]):
							add_parameter(rsm, _nb, _pb)

	return rsm, md

def lora2std(modin, types=tgt_types):

	return to_std(modin, types=types)
