#encoding: utf-8

import torch
from numpy import array as np_array, int32 as np_int32
from random import seed as rpyseed, shuffle

from utils.fmt.floader import Loader as LoaderBase
from utils.fmt.plm.custbert.raw.base import inf_file_loader
from utils.fmt.raw.cachepath import get_cache_fname, get_cache_path
from utils.fmt.raw.reader.sort.single import sort_lines_reader
from utils.fmt.single import batch_padder
from utils.fmt.vocab.char import ldvocab
from utils.fmt.vocab.plm.custbert import map_batch
from utils.h5serial import h5File
from utils.process import start_process

from cnfg.base import seed as rand_seed
from cnfg.ihyp import h5_fileargs, h5datawargs, max_pad_tokens_sentence, max_sentences_gpu, max_tokens_gpu, normal_tokens_vs_pad_tokens
from cnfg.vocab.plm.custbert import init_normal_token_id, init_vocab, pad_id, vocab_size

class Loader(LoaderBase):

	def __init__(self, sfiles, dfiles, vcbf, max_len=510, num_cache=2, raw_cache_size=4194304, skip_lines=0, nbatch=256, minfreq=False, vsize=vocab_size, ngpu=1, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, sleep_secs=1.0, file_loader=inf_file_loader, ldvocab=ldvocab, print_func=print, **kwargs):

		super(Loader, self).__init__(sleep_secs=sleep_secs, print_func=print_func, **kwargs)
		self.sent_files, self.doc_files, self.max_len, self.num_cache, self.raw_cache_size, self.skip_lines, self.nbatch, self.minbsize, self.maxpad, self.maxpart, self.file_loader = sfiles, dfiles, max_len, num_cache, raw_cache_size, skip_lines, nbatch, ngpu, maxpad, maxpart, file_loader
		self.bsize, self.maxtoken = (bsize, maxtoken,) if self.minbsize == 1 else (bsize * self.minbsize, maxtoken * self.minbsize,)
		self.cache_path = get_cache_path(*self.sent_files, *self.doc_files)
		self.vcb = ldvocab(vcbf, minf=minfreq, omit_vsize=vsize, vanilla=False, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id)[0]
		with self.todo_lck:
			self.todo.extend([get_cache_fname(self.cache_path, i=_) for _ in range(self.num_cache)])
		self.p_loader = start_process(target=self.loader)

	def loader(self):

		rpyseed(rand_seed)
		dloader = self.file_loader(self.sent_files, self.doc_files, max_len=self.max_len, print_func=None)
		file_reader = sort_lines_reader(line_read=self.raw_cache_size)
		if self.skip_lines > 0:
			_line_read = self.skip_lines - 1
			for _ind, _ in enumerate(dloader, 1):
				if _ind > _line_read:
					break
		while self.running.value:
			_cache_file = self.get_todo()
			if _cache_file is not None:
				with h5File(_cache_file, "w", **h5_fileargs) as rsf:
					src_grp = rsf.create_group("src")
					curd = 0
					for i_d in batch_padder(dloader, self.vcb, self.bsize, self.maxpad, self.maxpart, self.maxtoken, self.minbsize, file_reader=file_reader, map_batch=map_batch, pad_id=pad_id):
						src_grp.create_dataset(str(curd), data=np_array(i_d, dtype=np_int32), **h5datawargs)
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
			src_grp = td["src"]
			for i_d in tl:
				yield torch.from_numpy(src_grp[i_d][()])
			td.close()
			if self.print_func is not None:
				self.print_func("close %s" % _cache_file)
			with self.todo_lck:
				self.todo.append(_cache_file)
