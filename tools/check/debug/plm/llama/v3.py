#encoding: utf-8

import torch
import transformers

from transformer.PLM.LLaMa.v3.Decoder import Decoder as NMT
from utils.torch.comp import torch_inference_mode

import cnfg.plm.llama.v3.base as cnfg
from cnfg.plm.llama.v3.ihyp import *
from cnfg.vocab.plm.llama.v3 import vocab_size

llama_path = "plm/Llama-3.2-1B-Instruct"

disu = lambda a, b: (a.squeeze() - b.squeeze()).abs()
disf = lambda a, b: disu(a, b).sum()

print("Tokenization")
ht = transformers.PreTrainedTokenizerFast.from_pretrained(llama_path)#LlamaTokenizerFast
#print(ht.get_chat_template())
print(ht.tokenize("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHow are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"))
ids = ht.encode("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHow are you?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
print(ids)
#chat = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "How are you?"}]
#print(ht.apply_chat_template(chat, tokenize=False))
#ids = ht.apply_chat_template(chat, tokenize=True)
#print(ids)
ids = torch.as_tensor(ids, dtype=torch.long).unsqueeze(0)

print("load pre-trained models")
nm = NMT(cnfg.isize, vocab_size, cnfg.nlayer, fhsize=cnfg.ff_hsize, dropout=cnfg.drop, attn_drop=cnfg.attn_drop, act_drop=cnfg.act_drop, emb_w=None, num_head=cnfg.nhead, xseql=cache_len_default, ahsize=cnfg.attn_hsize, norm_output=cnfg.norm_output, bindemb=cnfg.bindDecoderEmb, num_kv_head=cnfg.kv_nhead, model_name=cnfg.model_name)
nm.load_plm("%s/model.h5" % llama_path)
nm.eval()

print("load models with transformers")
hm = transformers.LlamaForCausalLM.from_pretrained(llama_path)
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
	hm_gids = hm.generate(ids, max_length=512)
	nm_gids = nm.decode(ids, beam_size=1, max_len=512)
	nm_bgids = nm.decode(ids, beam_size=2, max_len=512, length_penalty=1.0)

print(ho)
print(no)
print(hm_gids)
print(nm_gids)
hdrs = ht.batch_decode(hm_gids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
ndrs = ht.batch_decode(nm_gids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
print(hdrs)
print(ndrs)
