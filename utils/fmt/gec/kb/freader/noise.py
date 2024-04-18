#encoding: utf-8

from utils.fmt.gec.kb.base import generate_iter_data
from utils.fmt.gec.noise.base import GECNoiseReader as GECNoiseReaderBase

from cnfg.hyp import cache_len_default

def gec_noise_reader_core(files=None, noiser=None, tokenizer=None, min_len=2, max_len=cache_len_default, **kwargs):

	_s_max_len = max_len - 2
	for _f in files:
		for line in _f:
			tgt = line.strip()
			if tgt:
				tgt = tgt.decode("utf-8")
				_l = len(tgt)
				if (_l > min_len) and (_l < _s_max_len):
					src = noiser(tgt)
					for _s, _k, _e, _t in generate_iter_data(tokenizer(src), None, tokenizer(tgt)):
						_l = len(_s)
						if _l < max_len:
							yield tuple(_s), tuple(_k), tuple(_e), tuple(_t)

def gec_noise_reader(fname=None, noiser=None, tokenizer=None, min_len=2, max_len=cache_len_default, inf_loop=False, gec_noise_reader_core=gec_noise_reader_core, **kwargs):

	return gec_noise_reader_base(fname=fname, noiser=noiser, tokenizer=tokenizer, min_len=min_len, max_len=max_len, inf_loop=inf_loop, gec_noise_reader_core=gec_noise_reader_core, **kwargs)

class GECNoiseReader(GECNoiseReaderBase):

	def __init__(self, fname, noiser, tokenizer, min_len=2, max_len=cache_len_default, inf_loop=False, gec_noise_reader=gec_noise_reader, gec_noise_reader_core=gec_noise_reader_core, **kwargs):

		super(GECNoiseReader, self).__init__(fname, noiser, tokenizer, min_len=min_len, max_len=max_len, inf_loop=inf_loop, gec_noise_reader=gec_noise_reader, gec_noise_reader_core=gec_noise_reader_core, **kwargs)
