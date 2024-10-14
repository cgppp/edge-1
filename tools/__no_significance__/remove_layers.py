#encoding: utf-8

""" usage:
	python remove_layers.py $src.h5 $rs.h5 enc/dec layers...
"""

import sys
from torch.nn import ModuleList

from transformer.NMT import NMT
from utils.base import remove_layers
from utils.h5serial import h5File
from utils.io import load_model_cpu, save_model

import cnfg.base as cnfg
from cnfg.ihyp import *

def handle(srcf, rsf, typ, rlist):

	with h5File(cnfg.dev_data, "r") as td:
		nword = td["nword"][()].tolist()
		nwordi, nwordt = nword[0], nword[-1]

	_tmpm = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes)
	_tmpm = load_model_cpu(srcf, _tmpm)
	if typ == "enc":
		_tmpm.enc.nets = ModuleList(remove_layers(list(_tmpm.enc.nets), rlist))
	elif typ == "dec":
		_tmpm.dec.nets = ModuleList(remove_layers(list(_tmpm.dec.nets), rlist))

	save_model(_tmpm, rsf, False)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], [int(_t) for _t in sys.argv[4:]])
