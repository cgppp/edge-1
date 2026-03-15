#encoding: utf-8
"""
将 HuggingFace 格式的 Qwen3 基座（safetensors 目录）转为本项目可加载的单个权重文件（.bin 或 .h5）。

用法:
  python tools/plm/convert_qwen3_hf_to_bin.py /home/common/plm/Qwen/Qwen3-0.6B-Base --output /path/to/model.bin
  python tools/plm/convert_qwen3_hf_to_bin.py /home/common/plm/Qwen/Qwen3-4B-Base --output /path/to/model.bin --format bin

要求: 已安装 transformers、torch；若输出 .h5 需能 import 本项目 cnfg（见下方）。
"""

import argparse
import os
import sys

def main():
	parser = argparse.ArgumentParser(description="Convert HuggingFace Qwen3 to project .bin/.h5")
	parser.add_argument("hf_dir", help="HuggingFace 模型目录，如 /home/common/plm/Qwen/Qwen3-0.6B-Base")
	parser.add_argument("--output", "-o", required=True, help="输出文件路径，如 model.bin 或 model.h5")
	parser.add_argument("--format", choices=("bin", "h5"), default=None, help="格式；不指定则根据 --output 扩展名推断")
	parser.add_argument("--no-save", action="store_true", help="仅检查 state_dict keys，不写入文件")
	args = parser.parse_args()

	if not os.path.isdir(args.hf_dir):
		sys.stderr.write("error: not a directory: %s\n" % args.hf_dir)
		sys.exit(1)

	fmt = args.format
	if fmt is None:
		if args.output.endswith(".h5"):
			fmt = "h5"
		elif args.output.endswith(".bin"):
			fmt = "bin"
		else:
			sys.stderr.write("error: cannot infer format from output path; use --format bin|h5\n")
			sys.exit(1)

	try:
		import torch
		from transformers import AutoModelForCausalLM
	except Exception as e:
		sys.stderr.write("error: need torch and transformers: %s\n" % e)
		sys.exit(1)

	print("Loading HuggingFace model from: %s" % args.hf_dir)
	model = AutoModelForCausalLM.from_pretrained(args.hf_dir, torch_dtype=torch.float32)
	sd = model.state_dict()
	print("state_dict keys (first 5): %s" % list(sd.keys())[:5])
	print("state_dict keys (last 3):  %s" % list(sd.keys())[-3:])
	if args.no_save:
		print("--no-save: skip writing")
		return

	if fmt == "bin":
		print("Saving to %s (PyTorch .bin) ..." % args.output)
		torch.save(sd, args.output)
		print("Done.")
		return

	# .h5: use project's h5save
	try:
		_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
		if _root not in sys.path:
			sys.path.insert(0, _root)
		from utils.h5serial import h5save
		from cnfg.ihyp import h5modelwargs
	except Exception as e:
		sys.stderr.write("error: for .h5 output need to run from project root and have cnfg: %s\n" % e)
		sys.exit(1)
	print("Saving to %s (HDF5) ..." % args.output)
	h5save(sd, args.output, h5args=h5modelwargs)
	print("Done.")


if __name__ == "__main__":
	main()
