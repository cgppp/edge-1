#encoding: utf-8

import sys

from transformer.Bern.NMT import NMT
from utils.base import report_parameters
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.prune import remove_maskq, report_prune_ratio

import cnfg.base as cnfg
from cnfg.ihyp import *

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

with h5File(cnfg.dev_data, "r", **h5_fileargs) as td:
	nword = td["nword"][()].tolist()
	nwordi, nwordt = nword[0], nword[-1]

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes)
mymodel.useBernMask(False)
mymodel = remove_maskq(mymodel)
mymodel = load_model_cpu(sys.argv[1], mymodel)
mymodel.apply(load_fixing)

print("Total parameter(s): %d" % (report_parameters(mymodel),))
for k, v in report_prune_ratio(mymodel).items():
	print("%s: %.2f" % (k, v * 100.0,))
