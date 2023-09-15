#encoding: utf-8

from random import choice, random
from re import compile, escape

from utils.fmt.parser import parse_none

def ordered_filter(lin):

	_ = set()
	for _lu in lin:
		if _lu and (_lu not in _):
			_.add(_lu)
			yield _lu

def re_many_or(opl, optm=False, re_escape=True, long_match=True, cache=None):

	if long_match:
		_ = {}
		for _xu in ordered_filter(opl):
			_l = len(_xu)
			if _l in _:
				_[_l].append(_xu)
			else:
				_[_l] = [_xu]
		_x = []
		for _l in sorted(_.keys(), reverse=True):
			_v = _[_l]
			if optm:
				_v.sort()
			_x.extend(_v)
	else:
		_x = list(ordered_filter(opl))
		if optm:
			_x.sort()
	if re_escape:
		_x = [escape(_) for _ in _x]
	_x = "|".join(_x)
	if isinstance(cache, dict):
		if _x in cache:
			return cache[_x]
		else:
			cache[_x] = _ = compile(_x)
			return _

	return compile(_x)

class SubMany:

	def __init__(self, dbl, *args, allow_re=False, **kwargs):

		self.re_escape = not allow_re
		self.set_ref(dbl)

	def handle(self, x):

		return self.data("", x)

	def set_ref(self, dbl):

		_ = re_many_or(set(dbl), optm=True, re_escape=self.re_escape, long_match=True).sub
		self.data = _

def compute_offset(sa, sb):

	lo = ro = None
	_ind = sb.find(sa)
	if _ind > -1:
		lo = -_ind
		ro = len(sb) + lo - len(sa)

	return lo, ro

class SubPair:

	def __init__(self, dbl, *args, allow_re=False, **kwargs):

		self.re_escape = not allow_re
		self.set_ref(dbl)

	def handle(self, x, offset=0):

		rs = x
		_sind = None
		if offset > 0:
			_m = self.search(rs[offset:])
			if _m is not None:
				_sind, _eind = _m.start() + offset, _m.end() + offset
		else:
			_m = self.search(rs)
			if _m is not None:
				_sind, _eind = _m.start(), _m.end()
		if _sind is not None:
			_k = rs[_sind:_eind]
			if _k in self.data:
				_v, _lo, _ro = self.data[_k]
				if (_lo is None) or (rs[_sind + _lo:_eind + _ro] != _v):
					_ = len(rs)
					rs = "%s%s%s" % (rs[:_sind], _v, rs[_eind:])
					if _eind < _:
						rs = self.handle(rs, offset=_eind + len(_v) - len(_k))

		return rs

	def set_ref(self, dbl, print_func=print):

		_d = {}
		for _k, _v in dbl:
			if isinstance(_k, str) and isinstance(_v, str):
				_lo, _ro = compute_offset(_k, _v)
				if (_k in _d) and (print_func is not None):
					print_func("%s: override %s -> %s" % (_k, _d[_k], _v,))
				_d[_k] = (_v, _lo, _ro,)
		_search = re_many_or(_d.keys(), optm=True, re_escape=self.re_escape, long_match=True).search
		self.data, self.search = _d, _search

class SubPairP:

	def __init__(self, dbl, *args, allow_re=False, p=0.1, **kwargs):

		self.re_escape, self.p = not allow_re, p
		self.set_ref(dbl)

	def handle(self, x, offset=0, p=None):

		rs = x
		_sind = None
		if offset > 0:
			_m = self.search(rs[offset:])
			if _m is not None:
				_sind, _eind = _m.start() + offset, _m.end() + offset
		else:
			_m = self.search(rs)
			if _m is not None:
				_sind, _eind = _m.start(), _m.end()
		if _sind is not None:
			_k = rs[_sind:_eind]
			if _k in self.data:
				_ = self.data[_k]
				_v, _lo, _ro = _[0] if len(_) == 1 else choice(_)
				if (_lo is None) or (rs[_sind + _lo:_eind + _ro] != _v):
					_p = parse_none(p, self.p)
					_ = len(rs)
					if random() < _p:
						rs = "%s%s%s" % (rs[:_sind], _v, rs[_eind:])
						if _eind < _:
							rs = self.handle(rs, offset=_eind + len(_v) - len(_k), p=_p)
					else:
						if _eind < _:
							rs = self.handle(rs, offset=_eind, p=_p)

		return rs

	def set_ref(self, dbl):

		_d = {}
		for _k, _v in dbl:
			if isinstance(_k, str) and isinstance(_v, str):
				_lo, _ro = compute_offset(_k, _v)
				if _k in _d:
					_d[_k].append((_v, _lo, _ro,))
				else:
					_d[_k] = [(_v, _lo, _ro,)]
		_search = re_many_or(_d.keys(), optm=True, re_escape=self.re_escape, long_match=True).search
		self.data, self.search = _d, _search
