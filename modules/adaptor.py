#encoding: utf-8

from torch import nn

from modules.base import PositionwiseFF as Adaptor
from utils.fmt.parser import parse_double_value_tuple

class IOAdaptor(nn.Module):

	def __init__(self, net, isize, hsize=None, dropout=0.0, act_drop=None, num_ia=1, use_decoding_cache=False, is_decoding=False, **kwargs):

		super(IOAdaptor, self).__init__()

		self.net = net
		_ihsize, _ohsize = parse_double_value_tuple(isize * 4 if hsize is None else hsize)
		self.i_adaptor = nn.ModuleList([Adaptor(isize, hsize=_ihsize, dropout=dropout, act_drop=act_drop, **kwargs) for _ in range(num_ia)])
		self.o_adaptor = Adaptor(isize, hsize=_ohsize, dropout=dropout, act_drop=act_drop, **kwargs)
		self.use_decoding_cache, self.is_decoding = use_decoding_cache, is_decoding
		self.decoding_cache = {}

	def forward(self, *args, **kwargs):

		if self.use_decoding_cache and self.is_decoding:
			_out = []
			for i, (_, _x,) in enumerate(zip(self.i_adaptor, args)):
				_i, _o = self.decoding_cache.get(i, (None, None,))
				if (_o is None) or (not _i.is_set_to(_x)):
					_o = _(_x)
					self.decoding_cache[i] = (_x, _o,)
				_out.append(_o)
		else:
			_out = [_(_x) for _, _x in zip(self.i_adaptor, args)]
		_ = len(_out)
		if len(args) > _:
			_out.extend(args[_:])

		_out = self.net(*_out, **kwargs)

		if isinstance(_out, tuple):
			return self.o_adaptor(_out[0]), *_out[1:]
		else:
			return self.o_adaptor(_out)

	def train(self, mode=True):

		super(IOAdaptor, self).train(mode)

		if mode:
			self.reset_buffer()
		self.is_decoding = not mode

		return self

	def reset_buffer(self, value=None):

		self.decoding_cache.clear()
