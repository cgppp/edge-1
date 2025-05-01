#encoding: utf-8

import sys
import torch
import transformers

from transformer.PLM.Baichuan.v2.Decoder import Decoder as NMT
from utils.torch.comp import torch_inference_mode

import cnfg.plm.baichuan.v2.base as cnfg
from cnfg.plm.baichuan.v2.ihyp import *
from cnfg.vocab.plm.baichuan.v2 import vocab_size

llm_path = "/home/common/plm/Baichuan/Baichuan2-7B-Chat"
sys.path.append(llm_path)

from generation_utils import build_chat_input
from modeling_baichuan import BaichuanForCausalLM
from tokenization_baichuan import BaichuanTokenizer

disu = lambda a, b: (a.squeeze() - b.squeeze()).abs()
disf = lambda a, b: disu(a, b).sum()

print("Tokenization")
ht = BaichuanTokenizer.from_pretrained(llm_path)

print("load models with transformers")
hm = BaichuanForCausalLM.from_pretrained(llm_path)
hm.generation_config = transformers.generation.utils.GenerationConfig.from_pretrained(llm_path)
hm.eval()

#chat = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "How are you?"}]
#print(build_chat_input(hm, ht, chat, max_new_tokens=512))
#ids = build_chat_input(hm, ht, chat, max_new_tokens=512)
#print(ids)
print(ht.tokenize("You are a helpful assistant.<reserved_106>How are you?<reserved_107>"))
ids = ht.encode("You are a helpful assistant.<reserved_106>How are you?<reserved_107>")
print(ids)
ids = torch.as_tensor(ids, dtype=torch.long).unsqueeze(0)

print("load pre-trained models")
nm = NMT(cnfg.isize, vocab_size, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, emb_w=None, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindemb=cnfg.bindDecoderEmb, model_name=cnfg.model_name)
nm.load_plm("%s/model.h5" % llm_path)
nm.eval()

with torch_inference_mode():
	print("Forward")
	ho = hm(ids).logits.log_softmax(-1)
	no = nm(ids)
	print("Build states")
	si = nm.build_states(ids)
	sp = nm.build_states(ids.narrow(1, 0, 4))
	sp = nm.build_states(ids.narrow(1, 4, ids.size(-1) - 4), states=sp)
	print("Inference")
	hm_gids = hm.generate(ids, max_new_tokens=512)
	nm_gids = nm.decode(ids, beam_size=1, max_len=512)
	#nm_bgids = nm.decode(ids, beam_size=2, max_len=512, length_penalty=1.0)

print(ho.squeeze())
print(no.squeeze())
print(hm_gids.squeeze())
print(nm_gids.squeeze())
hdrs = ht.batch_decode(hm_gids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
ndrs = ht.batch_decode(nm_gids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
print(hdrs)
print(ndrs)
