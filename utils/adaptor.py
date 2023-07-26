#encoding: utf-8

from torch.nn import ModuleList

from modules.adaptor import IOAdaptor

def share_ioadaptor_net(netin, ignore_num_ia=False):

	for net in netin.modules():
		if isinstance(net, ModuleList):
			net_d = {}
			for layer in net.modules():
				if isinstance(layer, IOAdaptor):
					_net_type = type(layer.net)
					_key = _net_type if ignore_num_ia else (_net_type, len(layer.i_adaptor),)
					if _key in net_d:
						layer.net = net_d[_key]
					else:
						net_d[_key] = layer.net

	return netin
