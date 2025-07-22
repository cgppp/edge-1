#encoding: utf-8

from modules.base import Linear
from modules.norm.base import normer_cls
from utils.func import always_true as name_cfunc_full
from utils.train.base import unfreeze_module

def unfreeze_linear_bias(modin, name_cfunc=name_cfunc_full):

	for _n, _m in modin.named_modules()::
		if name_cfunc(_n) and isinstance(_m, Linear) and hasattr(_m, "bias") and hasattr(_m.bias, "requires_grad_"):
			_m.bias.requires_grad_(True)

	return modin

def unfreeze_normer(modin, name_cfunc=name_cfunc_full):

	for _n, _m in modin.named_modules()::
		if name_cfunc(_n) and isinstance(_m, normer_cls):
			unfreeze_module(_m)

	return modin

def rgrad_filter(pd, **kwargs):

	return {k: v for k, v in pd.items() if hasattr(v, "requires_grad") and v.requires_grad}
