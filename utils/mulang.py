#encoding: utf-8

from math import ceil
from random import shuffle

from utils.data import inf_data_generator
from utils.random import multinomial

def T_normalize(wl, T):

	_t = 1.0 / T
	_mv = min(wl)
	_tmp = [(_wu / _mv) ** _t for _wu in wl]
	_s = sum(_tmp)

	return [_tu / _s for _tu in _tmp]

def sample_iter(wl, T, ntrain, taskl):

	samples = {}
	for i, (nd, task,) in enumerate(zip(ntrain, taskl)):
		samples[i] = (task, inf_data_generator(str(i) for i in range(nd)),)
	pl = T_normalize(wl, T)
	while True:
		task, dg = samples[multinomial(pl, s=1.0)]
		yield next(dg), task

class data_sampler:

	def __init__(self, task_weight, task_weight_T, ntrain, train_taskl, nsample=None, **kwargs):

		self.generator = sample_iter(task_weight, task_weight_T, ntrain, train_taskl)
		self.nsample = nsample

	def generate(self, nsample=None):

		return [next(self.generator) for i in range(self.nsample if nsample is None else nsample)]

class balance_loader:

	def __init__(self, tls, sfunc=min, **kwargs):

		self.tls = tls
		self.imax = len(tls)
		self.imin = - (self.imax + 1)
		self.ndata = self.imax * sfunc(len(_) for _ in self.tls)
		self.dg = [inf_data_generator(_) for _ in self.tls]
		self.c = [0 for _ in range(self.imax)]

	def get_one(self):

		_im, _vm = 0, self.c[0]
		for _i, _v in enumerate(self.c):
			if _v < _vm:
				_im, _vm = _i, _v

		return _im, next(self.dg[_im])

	def __call__(self, ndata=None, **kwargs):

		for _ in range(self.ndata if ndata is None else ndata):
			yield self.get_one()

	def update(self, i, v=0):

		if (i < self.imax) and (i > self.imin) and (v > 0):
			_ = self.c[i] + v
			self.c = [0 if _i == i else (_v - _) for _i, _v in enumerate(self.c)]

def sample_iter_token(tgen, taskl, tnt, tnpred, tdebit):

	for _t, _nt in zip(taskl, tnt):
		_npred, _g, _debit = tnpred[_t], tgen[_t], tdebit.get(_t, 0) + _nt
		while _debit > 0:
			i = next(_g)
			_debit -= _npred[i]
			yield i, _t
		tdebit[_t] = _debit

class data_sampler_token:

	def __init__(self, tnpredl, task_weight_T, ntrain, train_taskl, **kwargs):

		self.task_generators = {_t: inf_data_generator(str(_) for _ in range(_tntrain)) for _t, _tntrain in zip(train_taskl, ntrain)}
		self.train_taskl, self.tnpred, self.tdebit = train_taskl, {_k: {str(_i): _ for _i, _ in enumerate(_v)} for _k, _v in tnpredl.items()}, {}
		_ = [sum(tnpredl[_t]) for _t in train_taskl]
		_st = float(sum(_))
		self.tnt = [ceil(_w * _st) for _w in T_normalize([float(_nt) for _nt in _], task_weight_T)]

	def generate(self):

		_ = list(sample_iter_token(self.task_generators, self.train_taskl, self.tnt, self.tnpred, self.tdebit))
		shuffle(_)

		return _
