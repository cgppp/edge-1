#!/bin/bash

set -e -o pipefail -x

# this script is to build the h5 file for the single input case (lm), generation is taken care of by adv/predict/plm/predict_qwen.py

export cachedir=cache
export llmt=qwen/v3
export dataid=llm/w14de/$llmt

export tokenizer=/home/common/plm/Qwen/Qwen3-0.6B
export template=lm
export srcd=$cachedir/$dataid
export srctf=src.train.txt
export srcvf=src.dev.txt

export rsf_train=train.h5
export rsf_dev=dev.h5

export maxtokens=256

export ngpu=1

export do_sort=true
export do_map=true

export faext=".xz"

export wkd=$cachedir/$dataid

mkdir -p $wkd

export stif=$wkd/$srctf.ids$faext
export sdif=$wkd/$srcvf.ids$faext
if $do_map; then
	python tools/plm/map/$llmt.py $srcd/$srctf $tokenizer $stif $template &
	python tools/plm/map/$llmt.py $srcd/$srcvf $tokenizer $sdif $template &
	wait
fi

export stsf=$wkd/src.train.srt$faext
export sdsf=$wkd/src.dev.srt$faext
if $do_sort; then
	python tools/sort.py $stif $stsf $maxtokens &
	# use the following command to sort a very large dataset with limited memory
	#bash tools/lsort/sort.sh $stif $stsf $maxtokens &
	python tools/sort.py $sdif $sdsf 1048576 &
	wait
fi

python tools/plm/llmdec/mktest.py $stsf $wkd/$rsf_train $ngpu &
python tools/plm/llmdec/mktest.py $sdsf $wkd/$rsf_dev $ngpu &
wait
