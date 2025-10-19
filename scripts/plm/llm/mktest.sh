#!/bin/bash

set -e -o pipefail -x

export srcd=wmt14
export srctf=test.en
export modelf="expm/w14ed/std/base/merge.h5"
export rsd=w14trs
export rsf=$rsd/trans.txt

export cachedir=cache
export llmt=qwen/v3
export dataid=llm/w14de/$llmt

export tokenizer=/home/common/plm/Qwen/Qwen3-0.6B
export template=instruct_task

export ngpu=1

export sort_decode=true
export do_map=true

export faext=".xz"

export tgtd=$cachedir/$dataid

export bpef=out.bpe

mkdir -p $rsd

export stif=$tgtd/$srctf.ids$faext
if $do_map; then
	python tools/plm/map/$llmt.py $srcd/$srctf $tokenizer $stif $template
fi

if $sort_decode; then
	export srt_input_f=$tgtd/$srctf.srt$faext
	python tools/sort.py $stif $srt_input_f 1048576
else
	export srt_input_f=$stif
fi

python tools/plm/llmdec/mktest.py $srt_input_f $tgtd/test.h5 $ngpu
python predict.py $tgtd/$bpef.srt $tokenizer $modelf

if $sort_decode; then
	python tools/restore.py $stif $srt_input_f $tgtd/$bpef.srt $rsf
	rm $srt_input_f $tgtd/$bpef.srt
else
	mv $tgtd/$bpef.srt $rsf
fi

rm $stif $tgtd/test.h5
