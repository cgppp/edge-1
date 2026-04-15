#encoding: utf-8

import sys
from safetensors.torch import load_file, save_file

# h5File/h5load/h5write_dict: HDF5 读写封装
from utils.h5serial import h5File, h5load, h5write_dict

from cnfg.ihyp import *

def safetensors_to_h5(srcfl, rsf, h5args=h5zipargs):

	# 将一个或多个 safetensors 文件写入单个 HDF5
	# 典型用途：把 HuggingFace/训练产物的 .safetensors 转成项目内常用 .h5
	with h5File(rsf, "w", **h5_fileargs) as h5f:
		for srcf in srcfl:
			h5write_dict(h5f, load_file(srcf), h5args=h5args)

def h5_to_safetensors(srcfl, rsf):

	# 先读第一个 h5，再把后续 h5 的键值更新进去（后者同名键会覆盖前者）
	# 用于“多 h5 合并后导出 safetensors”
	_ = h5load(srcfl[0], restore_list=False)
	for srcf in srcfl[1:]:
		_.update(h5load(srcf, restore_list=False))
	save_file(_, rsf)

def handle(srcfl, rsf, **kwargs):

	# 根据输出后缀自动选择转换方向：
	# - rsf 以 .h5 结尾: safetensors -> h5
	# - 否则: h5 -> safetensors
	(safetensors_to_h5 if rsf.endswith(".h5") else h5_to_safetensors)(srcfl, rsf)

if __name__ == "__main__":
	# 用法:
	#   python tools/h5/convert/st.py in1 [in2 ...] out
	# 示例:
	#   python tools/h5/convert/st.py model.safetensors model.h5
	#   python tools/h5/convert/st.py a.h5 b.h5 merged.safetensors
	handle(sys.argv[1:-1], sys.argv[-1])
