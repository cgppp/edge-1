#encoding: utf-8

import sys
from numpy import array as np_array, int32 as np_int32, int8 as np_int8

from utils.fmt.gec.kb.quad import batch_padder
from utils.h5serial import h5File

from cnfg.ihyp import *

def handle(finput, fkb, fedit, ftarget, frs, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, minfreq=False, vsize=False):

	if expand_for_mulgpu:
		_bsize = bsize * minbsize
		_maxtoken = maxtoken * minbsize
	else:
		_bsize = bsize
		_maxtoken = maxtoken
	with h5File(frs, "w", **h5_fileargs) as rsf:
		src_grp = rsf.create_group("src")
		kb_grp = rsf.create_group("kb")
		edt_grp = rsf.create_group("edt")
		tgt_grp = rsf.create_group("tgt")
		curd = 0
		for i_d, kd, ed, td in batch_padder(finput, fkb, fedit, ftarget, _bsize, maxpad, maxpart, _maxtoken, minbsize):
			rid = np_array(i_d, dtype=np_int32)
			rkd = np_array(kd, dtype=np_int8)
			red = np_array(ed, dtype=np_int8)
			rtd = np_array(td, dtype=np_int32)
			wid = str(curd)
			src_grp.create_dataset(wid, data=rid, **h5datawargs)
			kb_grp.create_dataset(wid, data=rkd, **h5datawargs)
			edt_grp.create_dataset(wid, data=red, **h5datawargs)
			tgt_grp.create_dataset(wid, data=rtd, **h5datawargs)
			curd += 1
		rsf["ndata"] = np_array([curd], dtype=np_int32)
	print("Number of batches: %d" % curd)

if __name__ == "__main__":
	handle(*sys.argv[1:6], int(sys.argv[6]))
