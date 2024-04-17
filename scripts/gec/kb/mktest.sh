#!/bin/bash

set -e -o pipefail -x

export srcd=cache/kbgec
export srctf=src.test.txt
export kbtf=kb.test.txt
export modelf=""
export rsd=$srcd
export rsf=$rsd/pred.test.txt
export src_vcb=~/plm/custbert/char.vcb

export cachedir=cache
export dataid=kbgec

export ngpu=1

export sort_decode=true

export faext=".xz"

export tgtd=$cachedir/$dataid

export tgt_vcb=$src_vcb
export bpef=out.bpe

mkdir -p $rsd

export stif=$tgtd/$srctf.ids$faext
export ktif=$tgtd/$kbtf.ids$faext
python tools/plm/map/custbert.py $srcd/$srctf $src_vcb $stif &
python tools/plm/map/custbert.py $srcd/$kbtf $src_vcb $ktif &
wait

export smtif=$tgtd/$srctf.mids$faext
export kmtif=$tgtd/$kbtf.mids$faext
python tools/gec/kb/merge_src_kb.py $stif $ktif $smtif $kmtif

if $sort_decode; then
	export srt_input_f=$tgtd/$srctf.ids.srt$faext
	export srt_kb_f=$tgtd/$kbtf.ids.srt$faext
	python tools/sort.py $smtif $kmtif $srt_input_f $srt_kb_f 1048576
else
	export srt_input_f=$smtif
	export srt_kb_f=$kmtif
fi

python tools/gec/kb/mktest.py $srt_input_f $srt_kb_f $tgtd/test.h5 $ngpu
python predict_kbgec.py $tgtd/$bpef $tgt_vcb $modelf

if $sort_decode; then
	python tools/restore.py $smtif $kmtif $srt_input_f $srt_kb $tgtd/$bpef $rsf
	rm $srt_input_f $srt_kb $tgtd/$bpef
else
	mv $tgtd/$bpef $rsf
fi
rm $stif $ktif $smtif $kmtif $tgtd/test.h5
