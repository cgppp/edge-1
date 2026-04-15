#encoding: utf-8

import sys

# h5Dataset/h5File/h5FileList: 项目对 HDF5 的封装
# h5write_data: 将 numpy/tensor/标量等统一写回 HDF5
from utils.h5serial import h5Dataset, h5File, h5FileList, h5write_data
# merge_lora: 把 LoRA 低秩参数（A/B）并回基座权重，得到可直接推理/导出的完整权重
from utils.lora.para import merge_lora

from cnfg.ihyp import *

def handle(srcfl, rsf, h5args=h5zipargs, inplace=True, transpose=True, h5_fileargs=h5_fileargs, **kwargs):

	# d: 内存中的“参数名 -> 参数值”映射；会聚合多个输入文件后再统一 merge
	d = {}
	# srcfl 支持传多个 .h5（例如分片或基座+LoRA）；rsf 为输出 .h5
	# 读取阶段只收集参数，不立即写，便于做全量 merge
	print("lora.py: loading %s -> merge -> %s (no progress until done; large base may take many minutes)..." % (srcfl, rsf), flush=True)
	with h5FileList(srcfl, "r", **h5_fileargs) as fsrc, h5File(rsf, "w", **h5_fileargs) as frs:
		for _ in fsrc:
			for k, v in _.items():
				d[k] = v
		print("lora.py: read %d tensors, merging LoRA..." % len(d), flush=True)
		# 合并 LoRA 参数到主权重：
		# - inplace=True: 尽量原地更新减少内存占用
		# - transpose=True: 兼容本项目线性层权重布局（和 merge_lora 的实现约定一致）
		d = merge_lora(d, inplace=True, transpose=True)
		# 统一写出到目标 HDF5。若 v 还是 h5Dataset，需要先取 v[()] 变成可写实体
		print("lora.py: writing %d tensors to %s..." % (len(d), rsf), flush=True)
		for k, v in d.items():
			h5write_data(frs, k, v[()] if isinstance(v, h5Dataset) else v, h5args=h5args)
	print("lora.py: done.", flush=True)

if __name__ == "__main__":
	# 用法:
	#   python tools/h5/lora.py in1.h5 [in2.h5 ...] out.h5
	# 含义:
	#   读取一个或多个权重文件 -> merge LoRA -> 输出单个可部署/可推理权重文件
	handle(sys.argv[1:-1], sys.argv[-1])
