#!/bin/bash

# 失败即退出、管道中任一命令失败即退出、打印每条命令（便于排错）
set -e -o pipefail -x

# ---------- 任务输入与输出 ----------
# 原始测试源文件目录/文件
export srcd=wmt14
export srctf=test.en
# 模型文件（用于 predict.py）
export modelf="expm/w14ed/std/base/merge.h5"
# 最终翻译输出目录/文件
export rsd=w14trs
export rsf=$rsd/trans.txt

# ---------- 缓存与数据ID ----------
export cachedir=cache
export llmt=qwen/v3
export dataid=llm/w14de/$llmt

# tokenizer 与 prompt 模板
export tokenizer=/home/common/plm/Qwen/Qwen3-0.6B
export template=instruct_task

# mktest.py 中可按 ngpu 分片/并行处理测试集
export ngpu=1

# 是否做排序解码（先按长度排序，通常更快），以及是否先做文本->id映射
export sort_decode=true
export do_map=true

# 项目里中间缓存通常带压缩后缀（如 .xz）
export faext=".xz"

export tgtd=$cachedir/$dataid

# predict.py 生成的中间输出文件名（未restore前）
export bpef=out.bpe

mkdir -p $rsd

# stif: 源文本映射为 token id 后的中间文件
export stif=$tgtd/$srctf.ids$faext
if $do_map; then
	# 文本 -> tokenizer id；模板由 $template 控制
	python tools/plm/map/$llmt.py $srcd/$srctf $tokenizer $stif $template
fi

if $sort_decode; then
	# 按长度排序，减少 padding 浪费；后续要用 restore.py 还原原顺序
	export srt_input_f=$tgtd/$srctf.srt$faext
	python tools/sort.py $stif $srt_input_f 1048576
else
	export srt_input_f=$stif
fi

# 生成测试 h5（predict.py 会从 cnfg.test_data 读取该格式）
python tools/plm/llmdec/mktest.py $srt_input_f $tgtd/test.h5 $ngpu
# 执行推理，输出到排序后的中间结果文件
python predict.py $tgtd/$bpef.srt $tokenizer $modelf

if $sort_decode; then
	# 将排序后的输出恢复回原输入顺序
	python tools/restore.py $stif $srt_input_f $tgtd/$bpef.srt $rsf
	rm $srt_input_f $tgtd/$bpef.srt
else
	mv $tgtd/$bpef.srt $rsf
fi

# 清理中间文件，保留最终结果
rm $stif $tgtd/test.h5
