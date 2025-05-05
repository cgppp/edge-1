#encoding: utf-8

from utils.func import always_true as name_cfunc_full
from utils.torch.comp import torch_no_grad

def lora_filter(pd, lwset=set(["lora_wa", "lora_wb"])):

	return {k: v for k, v in pd.items() if k.endswith(".lora_wa") or k.endswith(".lora_wb") or (k in lwset)}

def merge_lora(pd, inplace=False, transpose=True, name_cfunc=name_cfunc_full):

	rs = {}
	_exs = set()
	with torch_no_grad():
		for n, p in pd.items():
			if (n.endswith(".weight") or (n == "weight")) and name_cfunc(n):
				_ = n[:-6]
				_lan, _lbn = "%slora_wa" % _, "%slora_wb" % _
				if (_lan in pd) and (_lbn in pd):
					_wa, _wb = pd[_lan], pd[_lbn]
					_ = _wa.mm(_wb)
					if transpose:
						_t = _.t()
						if _t.size() == p.size():
							_ = _t
						_t = None
					if _.size() == p.size():
						rs[n] = p.add_(_) if inplace else p.add(_)
						_exs.add(_lan)
						_exs.add(_lbn)
					else:
						rs[n] = p
					_ = None
					_wa = _wb = None
				else:
					rs[n] = p
			else:
				if n in _exs:
					_exs.remove(n)
				else:
					rs[n] = p
	if _exs:
		rs = {k: v for k, v in rs.items() if k not in _exs}

	return rs
