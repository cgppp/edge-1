#encoding: utf-8

from math import sqrt

from modules.lora.base import Linear
from utils.fmt.parser import parse_none
from utils.lora.ana.base import Ana

class GNormAna(Ana):

	def ana(self, model=None):

		rsd = {}
		_model = parse_none(model, self.model)
		if _model is not None:
			_an = 0.0
			for _name, _module in _model.named_modules():
				if isinstance(_module, Linear):
					#_ = sum(_.grad.pow(2.0).sum().item() for _ in _module.parameters() if _.requires_grad and (_.grad is not None))
					_ = _module.lora_wa.grad.pow(2.0).sum().item() + _module.lora_wb.grad.pow(2.0).sum().item()
					rsd[_name] = sqrt(_)
					_an += _
			rsd["."] = sqrt(_an)

		return rsd
