#encoding: utf-8

from json import dumps
from math import sqrt

from modules.lora import Linear
from utils.fmt.parser import parse_none

class GNormAna:

	def __init__(self, logf, model=None):

		self.logf, self.model = logf, model

	def compute_model_lora_norm(self, model=None):

		rsd = {}
		_an = 0.0
		for _name, _module in parse_none(model, self.model).named_modules():
			if isinstance(_module, Linear):
				#_ = sum(_.grad.pow(2.0).sum().item() for _ in _module.parameters() if _.requires_grad and (_.grad is not None))
				_ = _module.lora_wa.grad.pow(2.0).sum().item() + _module.lora_wb.grad.pow(2.0).sum().item()
				rsd[_name] = sqrt(_)
				_an += _
		rsd["."] = sqrt(_an)

		return rsd

	def __call__(self, model=None):

		_ = dumps(self.compute_model_lora_norm(model=model), ensure_ascii=False)
		with open(self.logf, "ab") as f:
			f.write(_.encode("utf-8"))
			f.write("\n".encode("utf-8"))
