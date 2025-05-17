#encoding: utf-8

import torch
from math import acos, pi, sqrt

from modules.lora.base import Linear
from utils.angle import prep_cos
from utils.fmt.parser import parse_none
from utils.lora.ana.base import Ana
from utils.torch.comp import torch_no_grad

from cnfg.ihyp import ieps_default

class WSimAna(Ana):

	def __init__(self, logf, model=None, **kwargs):

		super(WSimAna, self).__init__(logf, model=model, **kwargs)
		self.scale = 180.0 / pi
		if model is None:
			self.wdict = {}
		else:
			with torch_no_grad():
				self.wdict = {_name: [_module.lora_wa.clone(), _module.lora_wb.clone()] for _name, _module in model.named_modules() if isinstance(_module, Linear)}

	def ana(self, model=None, ieps=ieps_default):

		rsd = {}
		_model = parse_none(model, self.model)
		if _model is not None:
			ons = os = ns = None
			with torch_no_grad():
				for _name, _module in _model.named_modules():
					if isinstance(_module, Linear):
						if _name in self.wdict:
							on, o, n = zip(*[prep_cos(ou, nu) for ou, nu in zip(self.wdict[_name], [_module.lora_wa, _module.lora_wb])])
							_ons, _os, _ns = torch.stack(on, 0).sum().item(), torch.stack(o, 0).sum().item(), torch.stack(n, 0).sum().item()
							rsd[_name] = acos(min(max(-1.0, (_ons / (sqrt(_os * _ns) + ieps))), 1.0)) * self.scale
							if ons is None:
								ons, os, ns = _ons, _os, _ns
							else:
								ons += _ons
								os += _os
								ns += _ns
						self.wdict[_name] = [_module.lora_wa.clone(), _module.lora_wb.clone()]
			if ons is not None:
				rsd["."] = acos(min(max(-1.0, (ons / (sqrt(os * ns) + ieps))), 1.0)) * self.scale

		return rsd
