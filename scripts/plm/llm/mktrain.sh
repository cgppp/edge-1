#!/bin/bash

set -e -o pipefail -x

# use `from utils.fmt.plm.llmdec.dual import batch_padder` in `tools/plm/mkiodata.py`

export cachedir=cache
export llmt=qwen/v3
export dataid=llm/w14de/$llmt

export tokenizer=/home/common/plm/Qwen/Qwen3-0.6B
export template=instruct_task
export srcd=$cachedir/$dataid
export srctf=src.train.txt
export tgttf=tgt.train.txt
export srcvf=src.dev.txt
export tgtvf=tgt.dev.txt

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
export ttif=$wkd/$tgttf.ids$faext
export sdif=$wkd/$srcvf.ids$faext
export tdif=$wkd/$tgtvf.ids$faext
if $do_map; then
	python tools/plm/map/$llmt.py $srcd/$srctf $tokenizer $stif $template &
	python tools/plm/map/$llmt.py $srcd/$tgttf $tokenizer $ttif assistant &
	python tools/plm/map/$llmt.py $srcd/$srcvf $tokenizer $sdif $template &
	python tools/plm/map/$llmt.py $srcd/$tgtvf $tokenizer $tdif assistant &
	wait
fi

export stsf=$wkd/src.train.srt$faext
export ttsf=$wkd/tgt.train.srt$faext
export sdsf=$wkd/src.dev.srt$faext
export tdsf=$wkd/tgt.dev.srt$faext
if $do_sort; then
	python tools/sort.py $stif $ttif $stsf $ttsf $maxtokens &
	# use the following command to sort a very large dataset with limited memory
	#bash tools/lsort/sort.sh $stif $ttif $stsf $ttsf $maxtokens &
	python tools/sort.py $sdif $tdif $sdsf $tdsf 1048576 &
	wait
fi

python tools/plm/mkiodata.py $stsf $ttsf $wkd/$rsf_train $ngpu &
python tools/plm/mkiodata.py $sdsf $tdsf $wkd/$rsf_dev $ngpu &
wait
