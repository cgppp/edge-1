#encoding: utf-8

from utils.base import add_module
from utils.func import always_true as name_cfunc_full

def patch_std(modin, *args, type_pfunc={}, name_cfunc=name_cfunc_full, **kwargs):

	for _t, _f in type_pfunc.items():
		if isinstance(modin, _t):
			return _f(modin, *args, **kwargs), {".": modin}

	md = {}
	for _name, _module in modin.named_modules():
		if name_cfunc(_name):
			for _t, _f in type_pfunc.items():
				if isinstance(_module, _t):
					md[_name] = _module
					add_module(modin, _name, _f(_module, *args, **kwargs))

	return modin, md

def to_std(modin, types=()):

	if isinstance(modin, types) and hasattr(modin, "to_std"):
		return modin.to_std(), {".": modin}

	md = {}
	for _name, _module in modin.named_modules():
		if isinstance(_module, types) and hasattr(_module, "to_std"):
			md[_name] = _module
			_std_m = _module.to_std()
			add_module(modin, _name, _std_m)

	return modin, md

def restore_md(modin, md):

	if "." in md:
		return md["."]

	for _name, _module in md.items():
		add_module(modin, _name, _module)

	return modin
