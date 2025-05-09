#!/bin/bash

set -e -o pipefail -x

# use `from utils.fmt.plm.llmdec.dual import batch_padder` in `tools/plm/mkiodata.py`

export cachedir=cache
export dataid=llm/w14de

export rank_script=tools/check/eva/comet_score.py
export rank_model=/home/common/plm/COMET/wmt22-cometkiwi-da/checkpoints/model.ckpt
# get rank_thres by python $rank_script $srcd/$srcvf $srcd/$tgtvf $rank_model $dscoref $rank_ngpu $comet_cache_size
export rank_thres=0.8242031041388518
export rank_descend=1
export rank_ngpu=1
export filter_max_count=8

export llmt=qwen/v3
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

export get_rank_thres=true
export do_rank_sort=true
export do_rank_score=true
export do_rank_clean=true
export do_sort=true
export do_map=true
export do_filter=true

export faext=".xz"

export comet_cache_size=131072

export wkd=$cachedir/$dataid

mkdir -p $wkd/$llmt

export tscoref=$wkd/rank.train.scores$faext
export dscoref=$wkd/rank.dev.scores$faext

if $get_rank_thres; then
	python $rank_script $srcd/$srcvf $srcd/$tgtvf $rank_model $dscoref $rank_ngpu $comet_cache_size
	exit
fi

export strf=$wkd/src.train.rank.srt$faext
export ttrf=$wkd/tgt.train.rank.srt$faext
if $do_rank_sort; then
	python tools/sort.py $srcd/$srctf $srcd/$tgttf $strf $ttrf $maxtokens
	# use the following command to sort a very large dataset with limited memory
	#bash tools/lsort/sort.sh $srcd/$srctf $srcd/$tgttf $strf $ttrf $maxtokens
fi

if $do_rank_score; then
	python $rank_script $strf $ttrf $rank_model $tscoref $rank_ngpu $comet_cache_size
fi

export stcf=$wkd/src.train.clean$faext
export ttcf=$wkd/tgt.train.clean$faext
export rtcf=$wkd/rank.train.clean$faext
export stcsf=$wkd/src.train.clean.srt$faext
export ttcsf=$wkd/tgt.train.clean.srt$faext
if $do_rank_clean; then
	python tools/clean/rank.py $strf $ttrf $tscoref $tscoref $stcf $ttcf $rtcf $rank_thres $rank_descend
	python tools/sortby.py $stcf $ttcf $rtcf $stcsf $ttcsf $rank_descend
fi

export stif=$wkd/$llmt/$srctf.ids$faext
export ttif=$wkd/$llmt/$tgttf.ids$faext
export sdif=$wkd/$llmt/$srcvf.ids$faext
export tdif=$wkd/$llmt/$tgtvf.ids$faext
if $do_map; then
	python tools/plm/map/$llmt.py $stcsf $tokenizer $stif $template &
	python tools/plm/map/$llmt.py $ttcsf $tokenizer $ttif assistant &
	python tools/plm/map/$llmt.py $srcd/$srcvf $tokenizer $sdif $template &
	python tools/plm/map/$llmt.py $srcd/$tgtvf $tokenizer $tdif assistant &
	wait
fi

export stfif=$wkd/$llmt/$srctf.clean.ids$faext
export ttfif=$wkd/$llmt/$tgttf.clean.ids$faext
if $do_filter; then
	python tools/clean/maxcount.py $stif $ttif $stfif $ttfif $filter_max_count
fi

export stsf=$wkd/$llmt/src.train.srt$faext
export ttsf=$wkd/$llmt/tgt.train.srt$faext
export sdsf=$wkd/$llmt/src.dev.srt$faext
export tdsf=$wkd/$llmt/tgt.dev.srt$faext
if $do_sort; then
	python tools/sort.py $stfif $ttfif $stsf $ttsf $maxtokens &
	# use the following command to sort a very large dataset with limited memory
	#bash tools/lsort/sort.sh $stfif $ttfif $stsf $ttsf $maxtokens &
	python tools/sort.py $sdif $tdif $sdsf $tdsf 1048576 &
	wait
fi

python tools/plm/mkiodata.py $stsf $ttsf $wkd/$llmt/$rsf_train $ngpu &
python tools/plm/mkiodata.py $sdsf $tdsf $wkd/$llmt/$rsf_dev $ngpu &
wait
