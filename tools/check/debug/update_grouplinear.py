#encoding: utf-8

import sys
from torch import nn

from modules.group.base import GroupLinear
from transformer.MuLang.NMT import NMT
from utils.h5serial import h5File
from utils.io import load_model_cpu, save_model

import cnfg.mulang as cnfg
from cnfg.ihyp import *

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()
	if isinstance(module, GroupLinear) and (module.bias is not None) and (module.bias.dim() == 2):
		module.bias = nn.Parameter(module.bias.unsqueeze(1))

with h5File(cnfg.dev_data, "r", **h5_fileargs) as td:
	nword = td["nword"][()].tolist()
	nwordi, ntask, nwordt = nword[0], nword[1], nword[-1]

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes, ntask=ntask, ngroup=cnfg.ngroup)

mymodel = load_model_cpu(sys.argv[1], mymodel)
mymodel.apply(load_fixing)

save_model(mymodel, sys.argv[-1], False)
