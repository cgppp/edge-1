#encoding: utf-8

# usage: [CUDA_VISIBLE_DEVICES=""] python tools/check/llm/ht_gen.py $srcf $model_path $rsf

import sys
import torch
from json import dumps
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.fmt.base import sys_open

def handle(srcf, model_path, rsf, max_len=512, system="You are a helpful assistant."):

	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
	ens = "\n".encode("utf-8")
	with sys_open(srcf, "rb") as fsrc, sys_open(rsf, "wb") as fwrt:
		for _ in fsrc:
			_l = _.strip()
			if _l:
				_l = _l.decode("utf-8")
				chat = [{"role": "system", "content": system}] if system else []
				chat.append({"role": "user", "content": _l})
				ids = torch.as_tensor(tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True), dtype=torch.long, device=model.device).unsqueeze(0)
				gids = model.generate(ids, max_new_tokens=max_len)
				gids = gids.narrow(-1, ids.size(-1), gids.size(-1) - ids.size(-1)).squeeze()
				rs = tokenizer.decode(gids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
				fwrt.write(dumps(rs, ensure_ascii=False).encode("utf-8"))
			fwrt.write(ens)

if __name__ == "__main__":
	handle(*sys.argv[1:4])
