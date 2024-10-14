#encoding: utf-8

import sys
import torch

from parallel.parallelMT import DataParallelMT
from transformer.EnsembleNMT import NMT as Ensemble
from transformer.GECToR.KBNMT import NMT
from utils.base import set_random_seed
from utils.fmt.base import sys_open
from utils.fmt.base4torch import parse_cuda_decode
from utils.fmt.vocab.base import reverse_dict
from utils.fmt.vocab.char import ldvocab
from utils.h5serial import h5File
from utils.io import load_model_cpu
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode
from utils.tqdm import tqdm

import cnfg.gec.gector as cnfg
from cnfg.ihyp import *
from cnfg.vocab.plm.custbert import eos_id, init_normal_token_id, init_vocab, sos_id, vocab_size

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

td = h5File(cnfg.test_data, "r")

ntest = td["ndata"][()].item()
vcbt = reverse_dict(ldvocab(sys.argv[2], minf=False, omit_vsize=vocab_size, vanilla=False, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id)[0])

if len(sys.argv) == 4:
	mymodel = NMT(cnfg.isize, vocab_size, vocab_size, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes)
	mymodel.build_task_model(fix_init=False)

	mymodel = load_model_cpu(sys.argv[3], mymodel)
	mymodel.apply(load_fixing)

else:
	models = []
	for modelf in sys.argv[3:]:
		tmp = NMT(cnfg.isize, vocab_size, vocab_size, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, global_emb=cnfg.share_emb, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindDecoderEmb=cnfg.bindDecoderEmb, forbidden_index=cnfg.forbidden_indexes)
		tmp.build_task_model(fix_init=False)

		tmp = load_model_cpu(modelf, tmp)
		tmp.apply(load_fixing)

		models.append(tmp)
	mymodel = Ensemble(models)

mymodel.eval()

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda_decode(cnfg.use_cuda, cnfg.gpuid, cnfg.multi_gpu_decoding)
use_amp = cnfg.use_amp and use_cuda

set_random_seed(cnfg.seed, use_cuda)

if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)

mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)

beam_size = cnfg.beam_size
length_penalty = cnfg.length_penalty
op_keep_bias = cnfg.op_keep_bias
edit_thres = cnfg.edit_thres

ens = "\n".encode("utf-8")

src_grp, kb_grp = td["src"], td["kb"]
with sys_open(sys.argv[1], "wb") as f, torch_inference_mode():
	for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
		wid = str(i)
		seq_batch = torch.from_numpy(src_grp[wid][()])
		seq_kb = torch.from_numpy(kb_grp[wid][()])
		if cuda_device:
			seq_batch = seq_batch.to(cuda_device, non_blocking=True)
			seq_kb = seq_kb.to(cuda_device, non_blocking=True)
		seq_batch, seq_kb = seq_batch.long(), seq_kb.long()
		with torch_autocast(enabled=use_amp):
			output = mymodel.decode(seq_batch, kb=seq_kb, beam_size=beam_size, max_len=None, length_penalty=length_penalty, op_keep_bias=op_keep_bias, edit_thres=edit_thres)
		if multi_gpu:
			tmp = []
			for ou in output:
				tmp.extend(ou)
			output = tmp
		for tran in output:
			tmp = []
			_ = tran.tolist()
			if _[0] == sos_id:
				_ = _[1:]
			for tmpu in _:
				if tmpu == eos_id:
					break
				else:
					tmp.append(vcbt[tmpu])
			f.write("".join(tmp).encode("utf-8"))
			f.write(ens)

td.close()
