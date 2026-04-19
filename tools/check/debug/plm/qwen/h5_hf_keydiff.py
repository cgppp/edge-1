#encoding: utf-8
"""
对比 Qwen3「本仓库 model.h5」与 HuggingFace Qwen3ForCausalLM 的 state_dict 键名 / 形状。

- HF 侧：**不加载 safetensors 权重**，仅用 AutoConfig 建空模型取 `state_dict().keys()` 与 shape，省内存、秒开。
- H5 侧：用 h5py **只遍历 Dataset 路径**，不整文件 `h5load` 进内存，避免 8B 再占一份 RAM。

用法（项目根目录 + PYTHONPATH）：
  python tools/check/debug/plm/qwen/h5_hf_keydiff.py
  python tools/check/debug/plm/qwen/h5_hf_keydiff.py --llm-path /home/common/plm/Qwen/Qwen3-8B --model-h5 /path/to/model.h5
  python tools/check/debug/plm/qwen/h5_hf_keydiff.py --max-print 50
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import torch
from transformers import AutoConfig, Qwen3ForCausalLM


def _fix_flat_key(k: str) -> str:
	"""与 utils.fmt.plm.base.fix_parameter_name 对「单层键名」的等价处理（扁平路径）。"""
	if k.endswith(".gamma"):
		return k[:-6] + ".weight"
	if k.endswith(".beta"):
		return k[:-5] + ".bias"
	return k


def _h5_dataset_paths(h5_path: str) -> dict[str, tuple[int, ...]]:
	"""返回 {扁平点号键: shape}，仅 Dataset。"""
	out: dict[str, tuple[int, ...]] = {}

	def visitor(name: str, node: h5py.Dataset | h5py.Group) -> None:
		if isinstance(node, h5py.Dataset):
			flat = name.replace("/", ".")
			flat = _fix_flat_key(flat)
			out[flat] = tuple(node.shape)

	with h5py.File(h5_path, "r") as f:
		f.visititems(visitor)

	return out


def _hf_key_shapes(llm_path: str) -> dict[str, tuple[int, ...]]:
	config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)
	model = Qwen3ForCausalLM(config)
	return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def main() -> int:
	ap = argparse.ArgumentParser(description="model.h5 与 HF Qwen3 state_dict 键/形状对比")
	ap.add_argument(
		"--llm-path",
		default=os.environ.get("QWEN3_LLM_PATH", "/home/common/plm/Qwen/Qwen3-8B"),
	)
	ap.add_argument("--model-h5", default=None)
	ap.add_argument("--max-print", type=int, default=40, help="每类最多打印多少条")
	args = ap.parse_args()

	llm_path = args.llm_path.rstrip("/")
	model_h5 = args.model_h5 or os.path.join(llm_path, "model.h5")

	if not os.path.isfile(model_h5):
		print("错误: 找不到 model.h5:", model_h5, file=sys.stderr)
		return 1

	print("llm_path:", llm_path)
	print("model_h5:", model_h5)

	h5_ks = _h5_dataset_paths(model_h5)
	hf_ks = _hf_key_shapes(llm_path)

	set_h5 = set(h5_ks)
	set_hf = set(hf_ks)

	only_h5 = sorted(set_h5 - set_hf)
	only_hf = sorted(set_hf - set_h5)
	both = sorted(set_h5 & set_hf)

	shape_mismatch = []
	for k in both:
		if h5_ks[k] != hf_ks[k]:
			shape_mismatch.append((k, h5_ks[k], hf_ks[k]))

	print("\n======== 统计 ========")
	print("h5 参数项数:", len(set_h5))
	print("HF 参数项数:", len(set_hf))
	print("交集:", len(both))
	print("仅在 h5 中:", len(only_h5))
	print("仅在 HF 中:", len(only_hf))
	print("同名但 shape 不一致:", len(shape_mismatch))

	_mp = args.max_print
	if only_h5:
		print("\n--- 仅在 model.h5（前 %d 条）---" % _mp)
		for k in only_h5[:_mp]:
			print(" ", k, h5_ks[k])
		if len(only_h5) > _mp:
			print("  ... 共 %d 条" % len(only_h5))
	if only_hf:
		print("\n--- 仅在 HF state_dict（前 %d 条）---" % _mp)
		for k in only_hf[:_mp]:
			print(" ", k, hf_ks[k])
		if len(only_hf) > _mp:
			print("  ... 共 %d 条" % len(only_hf))
	if shape_mismatch:
		print("\n--- 同名 shape 不一致（前 %d 条）---" % _mp)
		for item in shape_mismatch[:_mp]:
			print(" ", item[0])
			print("     h5:", item[1], " hf:", item[2])
		if len(shape_mismatch) > _mp:
			print("  ... 共 %d 条" % len(shape_mismatch))

	# 非零退出：说明 load_plm 可能未覆盖或转换有误
	if only_hf or shape_mismatch:
		print("\n结论: HF 有而 h5 无、或 shape 不一致时，应优先检查权重导出与 load_plm 映射。", file=sys.stderr)
		return 2
	if only_h5:
		print("\n结论: 仅 h5 多出的键多为冗余；若数量极大再核对导出脚本。", file=sys.stderr)
		return 0
	print("\n结论: 键名与 shape 在交集上完全一致（空模型 HF 命名与 h5 一致）。")
	return 0


if __name__ == "__main__":
	sys.exit(main())
