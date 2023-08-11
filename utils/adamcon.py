#encoding: utf-8

from torch.nn import ModuleList

from modules.adamcon.base import AdamCon

def bind_adamcon(netin):

	for net in netin.modules():
		if isinstance(net, ModuleList):
			_share_module = None
			for layer in net.modules():
				if hasattr(layer, "adamcon") and isinstance(layer.adamcon, AdamCon):
					if _share_module is None:
						_share_module = layer.adamcon
					else:
						layer.adamcon = _share_module

	return netin
