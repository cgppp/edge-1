#encoding: utf-8

from utils.fmt.base import all_gt, all_lt
from utils.fmt.gec.kb.base import generate_iter_data
from utils.fmt.parser import parse_none

from cnfg.hyp import cache_len_default

def gec_noise_reader(loader=None, tokenizer=None, min_len=2, max_len=cache_len_default, **kwargs):

	_s_max_len = max_len - 2
	for _du in loader:
		_l = tuple(len(_) for _ in _du)
		if all_gt(_l, min_len) and all_lt(_l, _s_max_len):
			for _s, _k, _e, _t in generate_iter_data(*(tokenizer(_) for _ in _du)):
				_l = len(_s)
				if _l < max_len:
					yield tuple(_s), tuple(_k), tuple(_e), tuple(_t)

class GECNoiseReader:

	def __init__(self, loader, tokenizer, min_len=2, max_len=cache_len_default, gec_noise_reader=gec_noise_reader, **kwargs):

		self.loader, self.tokenizer, self.min_len, self.max_len, self.gec_noise_reader = loader, tokenizer, min_len, max_len, gec_noise_reader

	def __call__(self, loader=None, tokenizer=None, min_len=None, max_len=None, gec_noise_reader=None, **kwargs):

		return parse_none(gec_noise_reader, self.gec_noise_reader)(loader=parse_none(fname, self.loader), tokenizer=parse_none(tokenizer, self.tokenizer), min_len=parse_none(min_len, self.min_len), max_len=parse_none(max_len, self.max_len), **kwargs)
