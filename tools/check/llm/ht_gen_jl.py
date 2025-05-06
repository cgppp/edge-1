#encoding: utf-8

# usage: [CUDA_VISIBLE_DEVICES=""] python tools/check/llm/ht_gen_tl.py $srcf $model_path $rsf

import sys
import torch
from json import dumps, loads
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.base import set_random_seed
from utils.fmt.base import sys_open
from utils.fmt.base4torch import parse_cuda_decode
from utils.torch.comp import torch_inference_mode

import cnfg.base as cnfg

def handle(srcf, model_path, rsf, system="You are a helpful assistant.", load_chat=False, max_len=512, strip_last=True):

	use_cuda, cuda_device, cuda_devices, multi_gpu, use_amp, use_cuda_bfmp, use_cuda_fp16 = parse_cuda_decode(cnfg.use_cuda, gpuid=cnfg.gpuid, use_amp=cnfg.use_amp, multi_gpu_decoding=cnfg.multi_gpu_decoding, use_cuda_bfmp=cnfg.use_cuda_bfmp)
	set_random_seed(cnfg.seed, use_cuda)
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16 if use_cuda_bfmp else (torch.float16 if use_cuda_fp16 else torch.float32))
	model.eval()
	if use_cuda:
		model.to(cuda_device)
	ens = "\n".encode("utf-8")
	with sys_open(srcf, "rb") as fsrc, sys_open(rsf, "wb") as fwrt, torch_inference_mode():
		for _ in fsrc:
			_l = _.strip()
			if _l:
				if load_chat:
					chat = loads(_l.decode("utf-8"))
				else:
					_l = loads(_l.decode("utf-8"))
					chat = [{"role": "system", "content": system}] if system else []
					chat.append({"role": "user", "content": _l})
				ids = torch.as_tensor(tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True), dtype=torch.long, device=model.device).unsqueeze(0)
				gids = model.generate(ids, max_new_tokens=max_len)
				_sid = ids.size(-1)
				_len = gids.size(-1) - _sid
				if strip_last:
					_len -= 1
				gids = gids.narrow(-1, _sid, _len).squeeze().tolist()
				rs = tokenizer.decode(gids, skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
				fwrt.write(dumps(rs, ensure_ascii=False).encode("utf-8"))
			fwrt.write(ens)

if __name__ == "__main__":
	handle(*sys.argv[1:])
