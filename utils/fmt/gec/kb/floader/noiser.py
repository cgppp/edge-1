#encoding: utf-8

import torch
from numpy import array as np_array, int32 as np_int32, int8 as np_int8
from os.path import exists as fs_check
from random import seed as rpyseed, shuffle
from time import sleep

from utils.fmt.gec.kb.dynquad import batch_padder
from utils.fmt.gec.kb.freader.mixer import gec_noise_reader
from utils.fmt.gec.noise.floader import Loader as LoaderBase
from utils.fmt.kbegen.mploader import loader
from utils.fmt.raw.reader.sort.tag import sort_lines_reader
from utils.h5serial import h5File

from cnfg.gec.gector import noise_char, noise_vcb, plm_vcb, seed as rand_seed
from cnfg.ihyp import cache_len_default, h5_libver, h5datawargs, max_pad_tokens_sentence, max_sentences_gpu, max_tokens_gpu, normal_tokens_vs_pad_tokens

class Loader(LoaderBase):

	def __init__(self, sfile, vcbf=plm_vcb, noise_char=noise_char, noise_vcb=noise_vcb, max_len=cache_len_default, num_cache=2, raw_cache_size=4194304, minfreq=False, ngpu=1, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, sleep_secs=1.0, norm_u8=False, file_loader=gec_noise_reader, print_func=print, **kwargs):

		self.kbloader = loader()
		super(Loader, self).__init__(sfile, vcbf=vcbf, noise_char=noise_char, noise_vcb=noise_vcb, max_len=max_len, num_cache=num_cache, raw_cache_size=raw_cache_size, minfreq=minfreq, ngpu=ngpu, bsize=bsize, maxpad=maxpad, maxpart=maxpart, maxtoken=maxtoken, sleep_secs=sleep_secs, norm_u8=norm_u8, file_loader=file_loader, print_func=print_func, **kwargs)

	def loader(self):

		rpyseed(rand_seed)
		dloader = self.file_loader(self.kbloader, self.sfile, self.noiser, self.tokenizer, max_len=self.max_len)
		file_reader = sort_lines_reader(line_read=self.raw_cache_size)
		while self.running.value:
			if self.todo:
				_cache_file = self.todo.pop(0)
				with h5File(_cache_file, "w", libver=h5_libver) as rsf:
					src_grp = rsf.create_group("src")
					kb_grp = rsf.create_group("kb")
					edt_grp = rsf.create_group("edt")
					tgt_grp = rsf.create_group("tgt")
					curd = 0
					for i_d, kd, ed, td in batch_padder(dloader, self.bsize, self.maxpad, self.maxpart, self.maxtoken, self.minbsize, file_reader=file_reader):
						wid = str(curd)
						src_grp.create_dataset(wid, data=np_array(i_d, dtype=np_int32), **h5datawargs)
						kb_grp.create_dataset(wid, data=np_array(kd, dtype=np_int8), **h5datawargs)
						edt_grp.create_dataset(wid, data=np_array(ed, dtype=np_int8), **h5datawargs)
						tgt_grp.create_dataset(wid, data=np_array(td, dtype=np_int32), **h5datawargs)
						curd += 1
					rsf["ndata"] = np_array([curd], dtype=np_int32)
				self.out.append(_cache_file)
			else:
				sleep(self.sleep_secs)

	def iter_func(self, *args, **kwargs):

		while self.running.value and (not self.out):
			sleep(self.sleep_secs)
		if self.out:
			_cache_file = self.out.pop(0)
			if fs_check(_cache_file):
				try:
					td = h5File(_cache_file, "r")
				except Exception as e:
					td = None
					if self.print_func is not None:
						self.print_func(e)
				if td is not None:
					if self.print_func is not None:
						self.print_func("load %s" % _cache_file)
					tl = [str(i) for i in range(td["ndata"][()].item())]
					shuffle(tl)
					src_grp = td["src"]
					kb_grp = td["kb"]
					edt_grp = td["edt"]
					tgt_grp = td["tgt"]
					for i_d in tl:
						yield torch.from_numpy(src_grp[i_d][()]), torch.from_numpy(kb_grp[i_d][()]), torch.from_numpy(edt_grp[i_d][()]), torch.from_numpy(tgt_grp[i_d][()])
					td.close()
					if self.print_func is not None:
						self.print_func("close %s" % _cache_file)
			self.todo.append(_cache_file)
