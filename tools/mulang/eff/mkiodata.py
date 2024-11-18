#encoding: utf-8

import sys
from numpy import array as np_array, int32 as np_int32

from utils.fmt.mulang.eff.dual import batch_padder
from utils.fmt.vocab.token import ldvocab
from utils.h5serial import h5File

from cnfg.ihyp import *

def handle(finput, ftarget, fvocab_i, fvocab_t, fvocab_task, frs, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, minfreq=False, vsize=False):
	vcbi, nwordi = ldvocab(fvocab_i, minf=minfreq, omit_vsize=vsize, vanilla=False)
	vcbt, nwordt = ldvocab(fvocab_t, minf=minfreq, omit_vsize=vsize, vanilla=False)
	vcbtask, nwordtask = ldvocab(fvocab_task, minf=False, omit_vsize=False, vanilla=True)
	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	with h5File(frs, "w", **h5_fileargs) as rsf:
		curd = {}
		torder = []
		npred = []
		for i_d, td, taskd, npredb in batch_padder(finput, ftarget, vcbi, vcbt, vcbtask, _bsize, maxpad, maxpart, _maxtoken, minbsize):
			_str_taskd = str(taskd)
			if taskd in curd:
				task_grp = rsf[_str_taskd]
				src_grp = task_grp["src"]
				tgt_grp = task_grp["tgt"]
			else:
				task_grp = rsf.create_group(_str_taskd)
				src_grp = task_grp.create_group("src")
				tgt_grp = task_grp.create_group("tgt")
				if npred:
					rsf[str(torder[-1])].create_dataset("npred", data=np_array(npred, dtype=np_int32), **h5datawargs)
					npred.clear()
				torder.append(taskd)
			rid = np_array(i_d, dtype=np_int32)
			rtd = np_array(td, dtype=np_int32)
			_id = curd.get(taskd, 0)
			wid = str(_id)
			src_grp.create_dataset(wid, data=rid, **h5datawargs)
			tgt_grp.create_dataset(wid, data=rtd, **h5datawargs)
			npred.append(npredb)
			curd[taskd] = _id + 1
		if npred:
			rsf[_str_taskd].create_dataset("npred", data=np_array(npred, dtype=np_int32), **h5datawargs)
			npred.clear()
		rsf["taskorder"] = np_array(torder, dtype=np_int32)
		curd = [curd[_] for _ in torder]
		rsf["ndata"] = np_array(curd, dtype=np_int32)
		rsf["nword"] = np_array([nwordi, nwordtask, nwordt], dtype=np_int32)
	print("Number of Batches: %d\nSource Vocabulary Size: %d\nTarget Vocabulary Size: %d\nNumber of Tasks: %d" % (sum(curd), nwordi, nwordt, nwordtask,))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7]))
