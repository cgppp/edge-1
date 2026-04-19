#encoding: utf-8
"""
Qwen3-8B：本仓库 Decoder + model.h5 与 HuggingFace Qwen3ForCausalLM 对齐检查。

用法（在项目根目录、已激活 lora-edge 且 PYTHONPATH 含项目根）：
  python tools/check/debug/plm/qwen/v3_8b_align.py
  QWEN3_LLM_PATH=/path/to/Qwen3-8B python tools/check/debug/plm/qwen/v3_8b_align.py
  python tools/check/debug/plm/qwen/v3_8b_align.py --skip-generate   # 只做 forward，省显存
  python tools/check/debug/plm/qwen/v3_8b_align.py --device cpu      # 极慢，仅小模型或调试

检查内容：
  1) 同一段 input_ids 上 HF logits（log_softmax）与本地 nm forward 的 max/mean 绝对误差；
  2) 可选：短 greedy generate vs 本地 decode（beam=1）。
"""

from __future__ import annotations

import argparse
import os
import sys
import gc

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

from transformer.PLM.QWen.v3.Decoder import Decoder as NMT
from utils.torch.comp import cuda_support_bf16, torch_inference_mode

import cnfg.plm.qwen.v3.base as cnfg
from cnfg.plm.qwen.v3.ihyp import *
from cnfg.vocab.plm.qwen.v3 import vocab_size

# 与 cache 里 instruct_auto / NQ 测试前缀同风格，便于和 predict 输入一致
DEFAULT_PROMPT = (
	"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
	"<|im_start|>user\nHow are you?<|im_end|>\n"
	"<|im_start|>assistant\n"
)


def _pick_device(device_arg: str) -> torch.device:
	if device_arg == "cpu":
		return torch.device("cpu")
	if device_arg == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA 不可用，请改用 --device cpu")
		return torch.device("cuda", 0)
	# auto
	if torch.cuda.is_available():
		return torch.device("cuda", 0)
	return torch.device("cpu")


def _align_vocab(ho: torch.Tensor, no: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""最后一维按较小 vocab 截断，避免 padded vocab 宽度不一致。"""
	ld = min(ho.size(-1), no.size(-1))
	if ho.size(-1) != ld:
		ho = ho[..., :ld]
	if no.size(-1) != ld:
		no = no[..., :ld]
	return ho, no

def _cuda_gc() -> None:
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.ipc_collect()
	gc.collect()


def main() -> int:
	ap = argparse.ArgumentParser(description="Qwen3-8B HF vs 本地 Decoder 对齐")
	ap.add_argument(
		"--llm-path",
		default=os.environ.get("QWEN3_LLM_PATH", "/home/common/plm/Qwen/Qwen3-8B"),
		help="HuggingFace 模型目录（含 config、tokenizer、权重）",
	)
	ap.add_argument(
		"--model-h5",
		default=None,
		help="本仓库 load_plm 用的 .h5；默认 <llm-path>/model.h5",
	)
	ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
	ap.add_argument("--max-new-tokens", type=int, default=32, help="generate / decode 步数上限")
	ap.add_argument("--skip-generate", action="store_true", help="只做 forward 对齐，不跑生成")
	ap.add_argument(
		"--staged-gpu",
		action="store_true",
		help="GPU 下分阶段加载 HF/NM，降低显存峰值（推荐 8B 打开）",
	)
	args = ap.parse_args()

	llm_path = args.llm_path.rstrip("/")
	model_h5 = args.model_h5 or os.path.join(llm_path, "model.h5")
	device = _pick_device(args.device)

	print("llm_path:", llm_path)
	print("model_h5:", model_h5)
	print("device:", device)

	if not os.path.isfile(model_h5):
		print("错误: 找不到 model.h5:", model_h5, file=sys.stderr)
		return 1

	tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
	ids_list = tokenizer.encode(DEFAULT_PROMPT, add_special_tokens=False)
	ids = torch.as_tensor(ids_list, dtype=torch.long).unsqueeze(0)
	use_bf16 = device.type == "cuda" and cuda_support_bf16
	dtype = torch.bfloat16 if use_bf16 else (torch.float16 if device.type == "cuda" and not use_bf16 else torch.float32)

	# staged-gpu 下分两次上卡：先 HF 求 ho，再释放显存，再 NM 求 no，避免两份 8B 同驻显存。
	if device.type == "cuda" and args.staged_gpu:
		ids = ids.to(device, non_blocking=True)

	print("prompt token len:", ids.size(-1))
	if device.type == "cuda" and args.staged_gpu:
		# ----- 阶段 1：HF -----
		print("load Qwen3ForCausalLM (HF), dtype=", dtype)
		hm = Qwen3ForCausalLM.from_pretrained(llm_path, torch_dtype=dtype, trust_remote_code=True)
		hm.eval()
		hm.to(device, non_blocking=True)
		with torch_inference_mode():
			h_logits = hm(ids).logits.float()
			ho = h_logits.log_softmax(dim=-1).cpu()
			hm_out = None if args.skip_generate else hm.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False).cpu()
		del hm
		_cuda_gc()

		# ----- 阶段 2：本地 Decoder -----
		nm = NMT(
			cnfg.isize,
			vocab_size,
			cnfg.nlayer,
			fhsize=cnfg.ff_hsize,
			dropout=cnfg.drop,
			attn_drop=cnfg.attn_drop,
			act_drop=cnfg.act_drop,
			emb_w=None,
			num_head=cnfg.nhead,
			xseql=cache_len_default,
			ahsize=cnfg.attn_hsize,
			norm_output=cnfg.norm_output,
			bindemb=cnfg.bindDecoderEmb,
			num_kv_head=cnfg.kv_nhead,
			model_name=cnfg.model_name,
		)
		print("load_plm (本地)...")
		nm.load_plm(model_h5)
		nm.eval()
		nm_dtype = torch.bfloat16 if use_bf16 else torch.float16
		nm.to(device=device, dtype=nm_dtype, non_blocking=True)
		with torch_inference_mode():
			no = nm(ids).float().cpu()
			nm_out = None if args.skip_generate else nm.decode(ids, beam_size=1, max_len=args.max_new_tokens, ilen=None)
	else:
		# 原始路径（显存足够时可用）
		nm = NMT(
			cnfg.isize,
			vocab_size,
			cnfg.nlayer,
			fhsize=cnfg.ff_hsize,
			dropout=cnfg.drop,
			attn_drop=cnfg.attn_drop,
			act_drop=cnfg.act_drop,
			emb_w=None,
			num_head=cnfg.nhead,
			xseql=cache_len_default,
			ahsize=cnfg.attn_hsize,
			norm_output=cnfg.norm_output,
			bindemb=cnfg.bindDecoderEmb,
			num_kv_head=cnfg.kv_nhead,
			model_name=cnfg.model_name,
		)
		print("load_plm (本地)...")
		nm.load_plm(model_h5)
		nm.eval()
		if device.type == "cuda":
			nm_dtype = torch.bfloat16 if use_bf16 else torch.float16
			nm.to(device=device, dtype=nm_dtype, non_blocking=True)
		print("load Qwen3ForCausalLM (HF), dtype=", dtype)
		hm = Qwen3ForCausalLM.from_pretrained(llm_path, torch_dtype=dtype, trust_remote_code=True)
		hm.eval()
		if device.type == "cuda":
			hm.to(device, non_blocking=True)
		with torch_inference_mode():
			h_logits = hm(ids).logits.float()
			ho = h_logits.log_softmax(dim=-1)
			no = nm(ids).float()
			hm_out = None if args.skip_generate else hm.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
			nm_out = None if args.skip_generate else nm.decode(ids, beam_size=1, max_len=args.max_new_tokens, ilen=None)

	ho, no = _align_vocab(ho, no)
	diff = (ho - no).abs()
	print("---- log_softmax logits ----")
	print("shape HF:", tuple(ho.shape), "shape NM:", tuple(no.shape))
	print("max abs diff:", float(diff.max().cpu()))
	print("mean abs diff:", float(diff.mean().cpu()))

	if args.skip_generate:
		print("skip-generate: 结束。")
		return 0

	print("---- short generate (HF greedy vs NM greedy_decode) ----")

	# greedy_decode + post_ilen_rs：ilen is None 时为 batch 的 Python list[list[int]]
	if isinstance(nm_out, list) and nm_out and isinstance(nm_out[0], list):
		nm_ids_flat = nm_out[0]
	elif isinstance(nm_out, list) and nm_out and isinstance(nm_out[0], torch.Tensor):
		nm_ids_flat = nm_out[0].flatten().tolist()
	else:
		nm_ids_flat = torch.as_tensor(nm_out).flatten().tolist()

	hf_ids = hm_out[0].tolist() if isinstance(hm_out, torch.Tensor) else hm_out[0].tolist()
	hf_text = tokenizer.decode(hf_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
	nm_text = tokenizer.decode(nm_ids_flat, skip_special_tokens=False, clean_up_tokenization_spaces=False)
	print("HF:", hf_text[:800])
	print("NM:", nm_text[:800])

	return 0


if __name__ == "__main__":
	sys.exit(main())
