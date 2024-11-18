#encoding: utf-8

import torch
from numpy import array as np_array, int32 as np_int32, int8 as np_int8
from random import seed as rpyseed, shuffle

from utils.fmt.floader import Loader as LoaderBase
from utils.fmt.gec.noise.base import Noiser
from utils.fmt.gec.noise.freader import gec_noise_reader
from utils.fmt.gec.noise.triple import batch_padder
from utils.fmt.plm.custbert.token import Tokenizer
from utils.fmt.raw.cachepath import get_cache_fname, get_cache_path
from utils.fmt.raw.reader.sort.tag import sort_lines_reader
from utils.h5serial import h5File
from utils.process import start_process

from cnfg.gec.gector import noise_char, noise_vcb, plm_vcb, seed as rand_seed
from cnfg.ihyp import cache_len_default, h5_fileargs, h5datawargs, max_pad_tokens_sentence, max_sentences_gpu, max_tokens_gpu, normal_tokens_vs_pad_tokens

class Loader(LoaderBase):

	def __init__(self, sfile, vcbf=plm_vcb, noise_char=noise_char, noise_vcb=noise_vcb, max_len=cache_len_default, num_cache=2, raw_cache_size=4194304, minfreq=False, ngpu=1, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, sleep_secs=1.0, norm_u8=False, file_loader=gec_noise_reader, print_func=print, **kwargs):

		super(Loader, self).__init__(sleep_secs=sleep_secs, print_func=print_func, **kwargs)
		self.sfile, self.max_len, self.num_cache, self.raw_cache_size, self.minbsize, self.maxpad, self.maxpart, self.file_loader = sfile, max_len, num_cache, raw_cache_size, ngpu, maxpad, maxpart, file_loader
		self.bsize, self.maxtoken = (bsize, maxtoken,) if self.minbsize == 1 else (bsize * self.minbsize, maxtoken * self.minbsize,)
		self.cache_path = get_cache_path(self.sfile) if isinstance(self.sfile, str) else get_cache_path(*self.sfile)
		self.tokenizer = Tokenizer(vcbf, norm_u8=norm_u8)
		self.noiser = Noiser(char=noise_char, vcb=noise_vcb)
		with self.todo_lck:
			self.todo.extend([get_cache_fname(self.cache_path, i=_) for _ in range(self.num_cache)])
		self.p_loader = start_process(target=self.loader)

	def loader(self):

		rpyseed(rand_seed)
		dloader = self.file_loader(self.sfile, self.noiser, self.tokenizer, max_len=self.max_len, inf_loop=self.raw_cache_size is not None)
		file_reader = sort_lines_reader(line_read=self.raw_cache_size)
		while self.running.value:
			_cache_file = self.get_todo()
			if _cache_file is not None:
				with h5File(_cache_file, "w", **h5_fileargs) as rsf:
					src_grp, edt_grp, tgt_grp = rsf.create_group("src"), rsf.create_group("edt"), rsf.create_group("tgt")
					curd = 0
					for i_d, ed, td in batch_padder(dloader, self.bsize, self.maxpad, self.maxpart, self.maxtoken, self.minbsize, file_reader=file_reader):
						wid = str(curd)
						src_grp.create_dataset(wid, data=np_array(i_d, dtype=np_int32), **h5datawargs)
						edt_grp.create_dataset(wid, data=np_array(ed, dtype=np_int8), **h5datawargs)
						tgt_grp.create_dataset(wid, data=np_array(td, dtype=np_int32), **h5datawargs)
						curd += 1
					rsf["ndata"] = np_array([curd], dtype=np_int32)
				with self.out_lck:
					self.out.append(_cache_file)

	def iter_func(self, *args, **kwargs):

		td, _cache_file = self.get_h5()
		if td is not None:
			if self.print_func is not None:
				self.print_func("load %s" % _cache_file)
			tl = [str(i) for i in range(td["ndata"][()].item())]
			shuffle(tl)
			src_grp, edt_grp, tgt_grp = td["src"], td["edt"], td["tgt"]
			for i_d in tl:
				yield torch.from_numpy(src_grp[i_d][()]), torch.from_numpy(edt_grp[i_d][()]), torch.from_numpy(tgt_grp[i_d][()])
			td.close()
			if self.print_func is not None:
				self.print_func("close %s" % _cache_file)
			with self.todo_lck:
				self.todo.append(_cache_file)
