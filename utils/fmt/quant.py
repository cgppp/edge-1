#encoding: utf-8

from math import inf
from numbers import Integral

from utils.math import maxiv, miniv

is_legal_dim = lambda dim: (dim is None) or isinstance(dim, (Integral, str,))

def parse_dim(dim, data=None):

	rs = None
	if isinstance(dim, Integral):
		rs = dim
	if isinstance(dim, str) and (data is not None):
		_size = data.size()
		if sum((1 if _ > 1 else 0 for _ in _size)) > 1:
			rs = (maxiv(_size) if dim[-1] == "x" else miniv((_ if _ > 1 else inf for _ in _size)))[-1]

	return rs
