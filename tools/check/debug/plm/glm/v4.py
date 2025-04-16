#encoding: utf-8

import sys
import torch
import transformers

from transformer.PLM.GLM.v4.Decoder import Decoder as NMT
from utils.torch.comp import torch_inference_mode

import cnfg.plm.glm.v4.base as cnfg
from cnfg.plm.glm.v4.ihyp import *
from cnfg.vocab.plm.glm.v4 import vocab_size

llm_path = "/home/common/plm/GLM/GLM-4-9B-0414/"
#sys.path.append(llm_path)
#from tokenization_chatglm import ChatGLM4Tokenizer

disu = lambda a, b: (a.squeeze() - b.squeeze()).abs()
disf = lambda a, b: disu(a, b).sum()

print("Tokenization")
ht = transformers.PreTrainedTokenizerFast.from_pretrained(llm_path)#ChatGLM4Tokenizer("%s/tokenizer.model" % llm_path)
chat = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "How are you?"}]
print(ht.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
#ids = ht.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
#print(ids)
print(ht.tokenize("<|system|>\nYou are a helpful assistant.<|user|>\nHow are you?<|assistant|>\n"))
ids = ht.encode("<|system|>\nYou are a helpful assistant.<|user|>\nHow are you?<|assistant|>\n")
print(ids)
#print(ht.tokenize("\n\n"))
#print(ht.encode("\n\n"))
#print(ht.tokenize("\n"))
#print(ht.encode("\n"))
ids = torch.as_tensor(ids, dtype=torch.long).unsqueeze(0)

print("load pre-trained models")
nm = NMT(cnfg.isize, vocab_size, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, emb_w=None, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindemb=cnfg.bindDecoderEmb, num_kv_head=cnfg.kv_nhead, model_name=cnfg.model_name)
nm.load_plm("%s/model.h5" % llm_path)
nm.eval()

print("load models with transformers")
hm = transformers.GlmForCausalLM.from_pretrained(llm_path)
hm.eval()

with torch_inference_mode():
	print("Forward")
	ho = hm(ids).logits.log_softmax(-1)
	no = nm(ids)
	print("Build states")
	#si = nm.build_states(ids)
	#sp = nm.build_states(ids.narrow(1, 0, 4))
	#sp = nm.build_states(ids.narrow(1, 4, ids.size(-1) - 4), states=sp)
	print("Inference")
	hm_gids = hm.generate(ids, max_new_tokens=128)
	hm_gids = hm_gids.narrow(-1, ids.size(-1), hm_gids.size(-1) - ids.size(-1))
	nm_gids = nm.decode(ids, beam_size=1, max_len=128)
	#nm_bgids = nm.decode(ids, beam_size=2, max_len=512, length_penalty=1.0)

print(ho.squeeze())
print(no.squeeze())
print(hm_gids.squeeze())
print(nm_gids.squeeze())
hdrs = ht.batch_decode(hm_gids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
ndrs = ht.batch_decode(nm_gids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
print(hdrs)
print(ndrs)
