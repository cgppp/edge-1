#encoding: utf-8

from numbers import Number

from utils.fmt.gec.kb.freader.kb import gec_noise_reader as kb_gec_noise_reader
from utils.fmt.gec.kb.freader.noise import gec_noise_reader as noise_gec_noise_reader
from utils.fmt.parser import parse_none
from utils.math import miniv

from cnfg.hyp import cache_len_default

def gec_noise_reader(loader=None, fname=None, noiser=None, tokenizer=None, min_len=2, max_len=cache_len_default, weight=None, **kwargs):

	_readers = {0: kb_gec_noise_reader(loader=loader, tokenizer=tokenizer, min_len=min_len, max_len=max_len, **kwargs), 1: noise_gec_noise_reader(fname=fname, noiser=noiser, tokenizer=tokenizer, min_len=min_len, max_len=max_len, inf_loop=True, **kwargs)}
	_w = None if weight is None else ({0: weight} if isinstance(weight, Number) else {_i: _ for _i, _ in enumerate(weight)})
	_n = [0.0 for _ in range(len(_readers))]
	while True:
		_m, _i = miniv(_n)
		_ = next(_readers[_i])
		yield _
		_acc = float(len(_[0]))
		if _w is not None:
			_acc *= _w.get(_i, 1.0)
		_n = [_acc if _ == _i else (_v - _m) for _, _v in enumerate(_n)]

class GECNoiseReader:

	def __init__(self, loader, fname, noiser, tokenizer, min_len=2, max_len=cache_len_default, weight=None, gec_noise_reader=gec_noise_reader, **kwargs):

		self.loader, self.fname, self.noiser, self.tokenizer, self.min_len, self.max_len, self.weight, self.gec_noise_reader = loader, fname, noiser, tokenizer, min_len, max_len, weight, gec_noise_reader

	def __call__(self, loader=None, fname=None, noiser=None, tokenizer=None, min_len=None, max_len=None, weight=None, gec_noise_reader=None, **kwargs):

		return parse_none(gec_noise_reader, self.gec_noise_reader)(loader=parse_none(fname, self.loader), fname=parse_none(fname, self.fname), noiser=parse_none(noiser, self.noiser), tokenizer=parse_none(tokenizer, self.tokenizer), min_len=parse_none(min_len, self.min_len), max_len=parse_none(max_len, self.max_len), weight=parse_none(weight, self.weight), **kwargs)
