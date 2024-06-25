#encoding: utf-8

from itertools import accumulate
from math import inf, log

from cnfg.ihyp import math_prefer_tail

def pos_norm(x):

	_s = sum(x)
	if _s == 0.0:
		_s = 1.0

	return [_ / _s for _ in x]

def cumsum(*args, **kwargs):

	return list(accumulate(*args, **kwargs))

def arcsigmoid(x):

	return -log((1.0 / x) - 1.0)

def miniv_h(x):

	_minv = inf
	_ind = 0
	for _i, _v in enumerate(x):
		if _v < _minv:
			_ind = _i
			_minv = _v

	return _minv, _ind

def miniv_t(x):

	_minv = inf
	_ind = 0
	for _i, _v in enumerate(x):
		if _v <= _minv:
			_ind = _i
			_minv = _v

	return _minv, _ind

def maxiv_h(x):

	_minv = -inf
	_ind = 0
	for _i, _v in enumerate(x):
		if _v > _minv:
			_ind = _i
			_minv = _v

	return _minv, _ind

def maxiv_t(x):

	_minv = -inf
	_ind = 0
	for _i, _v in enumerate(x):
		if _v >= _minv:
			_ind = _i
			_minv = _v

	return _minv, _ind

miniv, maxiv = (miniv_t, maxiv_t) if math_prefer_tail else (miniv_h, maxiv_h)

def exp_grow(start, end, k):

	_ng = k - 1
	_factor = (end / start) ** (1.0 / _ng)
	tmp = start
	rs = [start]
	for i in range(_ng):
		tmp *= _factor
		rs.append(tmp)

	return rs

def linear_grow(start, end, k):

	_ng = k - 1
	_factor = (end - start) / _ng
	tmp = start
	rs = [start]
	for i in range(_ng):
		tmp += _factor
		rs.append(tmp)

	return rs

def comb_grow(start, end, k, alpha=0.5):

	beta = 1.0 - alpha

	return [a * alpha + b * beta for a, b in zip(exp_grow(start, end, k), linear_grow(start, end, k))]

mvavg_dist2beta = lambda d: float(d) / float(d + 1)
