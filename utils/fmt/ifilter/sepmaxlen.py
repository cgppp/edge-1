#encoding: utf-8

from math import ceil, floor

def ifilter(x, *args, max_len=None, **kwargs):

	for _ in x:
		_l = len(_)
		if _l <= max_len:
			yield _
		else:
			_fl = float(_l)
			_lu = floor(_fl / ceil(_fl / max_len))
			_sid = 0
			while _sid < _l:
				_eid = min(_l, _sid + _lu)
				yield _[_sid:_eid]
				_sid = _eid
