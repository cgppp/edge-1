#encoding: utf-8

import sys

from transformer.NMT import NMT as NMTBase
from transformer.Probe.NMT import NMT
from utils.h5serial import h5File
from utils.init.base import init_model_params
from utils.io import load_model_cpu, save_model

import cnfg.probe as cnfg
from cnfg.ihyp import *

def handle(cnfg, srcmtf, decf, rsf):

	with h5File(cnfg.dev_data, "r") as tdf:
		nwordi, nwordt = tdf["nword"][()].tolist()

	mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes, num_layer_ana=cnfg.num_layer_fwd)
	init_model_params(mymodel)
	_tmpm = NMTBase(cnfg.isize, nwordi, nwordt, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes)
	_tmpm = init_model_params(_tmpm)
	_tmpm = load_model_cpu(srcmtf, _tmpm)
	mymodel.load_base(_tmpm)
	mymodel.dec = load_model_cpu(decf, mymodel.dec)
	if cnfg.share_emb:
		mymodel.dec.wemb.weight = _tmpm.enc.wemb.weight
	if cnfg.bindDecoderEmb:
		mymodel.dec.classifier.weight = mymodel.dec.wemb.weight
	_tmpm = None

	save_model(mymodel, rsf, sub_module=False, h5args=h5zipargs)

if __name__ == "__main__":
	handle(cnfg, sys.argv[1], sys.argv[2], sys.argv[3])
