#encoding: utf-8

import torch
import transformers

from transformer.PLM.Gemma.v3.Decoder import Decoder as NMT
from utils.norm.mp.f import convert as make_mp_model
from utils.torch.comp import torch_inference_mode

import cnfg.plm.gemma.v3.base as cnfg
from cnfg.plm.gemma.v3.ihyp import *
from cnfg.vocab.plm.gemma.v3 import vocab_size

llm_path = "/home/common/plm/Gemma/gemma-3-1b-it"
use_cuda_bfmp = False

disu = lambda a, b: (a.squeeze() - b.squeeze()).abs()
disf = lambda a, b: disu(a, b).sum()

def make_batch(*args):

	rs = []
	ilen = []
	_l = max(len(_) for _ in args)
	for _ in args:
		_cl = len(_)
		rs.append(_ + [0 for _tmp in range(_l - _cl)])
		ilen.append(_cl)

	return rs, ilen

print("Tokenization")
ht = transformers.GemmaTokenizerFast.from_pretrained(llm_path)
#print(ht.get_chat_template())
chat = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "How are you?"}]
print(ht.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
ids = ht.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
print(ids)
print(ht.tokenize("<start_of_turn>user\nYou are a helpful assistant.\n\nHow are you?<end_of_turn>\n<start_of_turn>model\n"))
ids = ht.encode("<start_of_turn>user\nYou are a helpful assistant.\n\nHow are you?<end_of_turn>\n<start_of_turn>model\n")
print(ids)
#idsl = ht.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi, how are you?<|im_end|>\n<|im_start|>assistant\n")
#print(idsl)
#idsb, ilen = make_batch(ids, idsl)
ids = torch.as_tensor(ids, dtype=torch.long).unsqueeze(0)
#idsl = torch.as_tensor(idsl, dtype=torch.long).unsqueeze(0)
#idsb = torch.as_tensor(idsb, dtype=torch.long)
#ilen = torch.as_tensor(ilen, dtype=torch.long)
#print(idsb)
#print(ilen)

print("load pre-trained models")
nm = NMT(cnfg.isize, vocab_size, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, emb_w=None, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindemb=cnfg.bindDecoderEmb, num_kv_head=cnfg.kv_nhead, model_name=cnfg.model_name)
#if use_cuda_bfmp:
#	make_mp_model(nm)
nm.load_plm("%s/model.h5" % llm_path)
nm.eval()

print("load models with transformers")
hm = transformers.Gemma3ForCausalLM.from_pretrained(llm_path)
hm.eval()

with torch_inference_mode():
	print("Forward")
	ho = hm(ids).logits.log_softmax(-1)
	no = nm(ids)
	print("Build states")
	si = nm.build_states(ids)
	sp = nm.build_states(ids.narrow(1, 0, 4))
	sp = nm.build_states(ids.narrow(1, 4, ids.size(-1) - 4), states=sp)
	print("Inference")
	hm_gids = hm.generate(ids, max_new_tokens=128)
	hm_gids = hm_gids.narrow(-1, ids.size(-1), hm_gids.size(-1) - ids.size(-1))
	nm_gids = nm.decode(ids, beam_size=1, max_len=128, length_penalty=0.0, repetition_penalty=1.0)
	#nm_gidsl = nm.decode(idsl, beam_size=1, max_len=128, length_penalty=0.0, repetition_penalty=1.0)
	#nm_gidsb = nm.decode(idsb, beam_size=1, max_len=128, length_penalty=0.0, repetition_penalty=1.0, ilen=ilen)
	#nm_bgids = nm.decode(ids, beam_size=3, max_len=512, length_penalty=0.0)

print(ho.squeeze())
print(no.squeeze())
print(hm_gids.squeeze())
print(nm_gids.squeeze())
hdrs = ht.batch_decode(hm_gids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
ndrs = [ht.decode(_, skip_special_tokens=False, clean_up_tokenization_spaces=False) for _ in nm_gids]
#ndrsl = [ht.decode(_, skip_special_tokens=False, clean_up_tokenization_spaces=False) for _ in nm_gidsl]
#ndrsb = [ht.decode(_, skip_special_tokens=False, clean_up_tokenization_spaces=False) for _ in nm_gidsb]
#print(nm_gids.squeeze())
#print(nm_gidsl.squeeze())
#print(nm_gidsb)
print(hdrs)
print(ndrs)
#print(ndrsl)
#print(ndrsb)
