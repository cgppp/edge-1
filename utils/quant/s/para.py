#encoding: utf-8

from modules.quant.s.base import QPara

def get_named_parameters(modin, prefix="", remove_duplicate=True, include_qpara=True):

	for n, p in modin.named_parameters(prefix=prefix, remove_duplicate=remove_duplicate):
		yield n, p
	if include_qpara:
		for n, m in modin.named_modules(prefix=prefix, remove_duplicate=remove_duplicate):
			if isinstance(m, QPara):
				yield, n, m.dequant()
