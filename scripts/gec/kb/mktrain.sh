#!/bin/bash

set -e -o pipefail -x

export cachedir=cache
export dataid=gec/kb/noise

export srcd=$cachedir/$dataid
export srctf=src.train.txt
export kbtf=kb.train.txt
export tgttf=tgt.train.txt
export srcvf=src.dev.txt
export kbvf=kb.dev.txt
export tgtvf=tgt.dev.txt
export rsrctf=raw.src.train.txt
export rtgttf=raw.tgt.train.txt
export rsrcvf=raw.src.dev.txt
export rtgtvf=raw.tgt.dev.txt
export src_vcb=~/plm/custbert/char.vcb

export rsf_train=train.h5
export rsf_dev=dev.h5

export maxtokens=512

export ngpu=1

export do_map=true
export do_sort=true

export faext=".xz"

export wkd=$cachedir/$dataid

mkdir -p $wkd

export stif=$wkd/$srctf.ids$faext
export ktif=$wkd/$kbtf.ids$faext
export ttif=$wkd/$tgttf.ids$faext
export sdif=$wkd/$srcvf.ids$faext
export kdif=$wkd/$kbvf.ids$faext
export tdif=$wkd/$tgtvf.ids$faext
export rstif=$wkd/$rsrctf.ids$faext
export rttif=$wkd/$rtgttf.ids$faext
export rsdif=$wkd/$rsrcvf.ids$faext
export rtdif=$wkd/$rtgtvf.ids$faext
export stcf=$wkd/src.train.ids$faext
export ktcf=$wkd/kb.train.ids$faext
export etcf=$wkd/edit.train.ids$faext
export ttcf=$wkd/tgt.train.ids$faext
export sdcf=$wkd/src.dev.ids$faext
export kdcf=$wkd/kb.dev.ids$faext
export edcf=$wkd/edit.dev.ids$faext
export tdcf=$wkd/tgt.dev.ids$faext
export rstcf=$wkd/raw.src.train.ids$faext
export rktcf=$wkd/raw.kb.train.ids$faext
export retcf=$wkd/raw.edit.train.ids$faext
export rttcf=$wkd/raw.tgt.train.ids$faext
export rsdcf=$wkd/raw.src.dev.ids$faext
export rkdcf=$wkd/raw.kb.dev.ids$faext
export redcf=$wkd/raw.edit.dev.ids$faext
export rtdcf=$wkd/raw.tgt.dev.ids$faext
if $do_map; then
	python tools/plm/map/custbert.py $srcd/$srctf $src_vcb $stif &
	python tools/plm/map/custbert.py $srcd/$kbtf $src_vcb $ktif &
	python tools/plm/map/custbert.py $srcd/$tgttf $src_vcb $ttif &
	python tools/plm/map/custbert.py $srcd/$srcvf $src_vcb $sdif &
	python tools/plm/map/custbert.py $srcd/$kbvf $src_vcb $kdif &
	python tools/plm/map/custbert.py $srcd/$tgtvf $src_vcb $tdif &
	python tools/plm/map/custbert.py $srcd/$rsrctf $src_vcb $rstif &
	python tools/plm/map/custbert.py $srcd/$rtgttf $src_vcb $rttif &
	python tools/plm/map/custbert.py $srcd/$rsrcvf $src_vcb $rsdif &
	python tools/plm/map/custbert.py $srcd/$rtgtvf $src_vcb $rtdif &
	wait
	python tools/gec/kb/convert.py $stif $ktif $ttif $stcf $ktcf $etcf $ttcf &
	python tools/gec/kb/convert.py $sdif $kdif $tdif $sdcf $kdcf $edcf $tdcf &
	python tools/gec/kb/convert_wokb.py $rstif $rttif $rstcf $rktcf $retcf $rttcf &
	python tools/gec/kb/convert_wokb.py $rsdif $rtdif $rsdcf $rkdcf $redcf $rtdcf &
	wait
	cat $rstcf >> $stcf &
	cat $rktcf >> $ktcf &
	cat $retcf >> $etcf &
	cat $rttcf >> $ttcf &
	cat $rsdcf >> $sdcf &
	cat $rkdcf >> $kdcf &
	cat $redcf >> $edcf &
	cat $rtdcf >> $tdcf &
	wait
	rm $rstcf $rktcf $retcf $rttcf $rsdcf $rkdcf $redcf $rtdcf &
fi

export stsf=$wkd/src.train.srt$faext
export ktsf=$wkd/kb.train.srt$faext
export etsf=$wkd/edit.train.srt$faext
export ttsf=$wkd/tgt.train.srt$faext
export sdsf=$wkd/src.dev.srt$faext
export kdsf=$wkd/kb.dev.srt$faext
export edsf=$wkd/edit.dev.srt$faext
export tdsf=$wkd/tgt.dev.srt$faext
if $do_sort; then
	python tools/sort.py $stcf $ktcf $etcf $ttcf $stsf $ktsf $etsf $ttsf $maxtokens &
	# use the following command to sort a very large dataset with limited memory
	#bash tools/lsort/sort.sh $stcf $ktcf $etcf $ttcf $stsf $ktsf $etsf $ttsf $maxtokens &
	python tools/sort.py $sdcf $kdcf $edcf $tdcf $sdsf $kdsf $edsf $tdsf 1048576 &
	wait
fi

python tools/gec/kb/mkiodata.py $stsf $ktsf $etsf $ttsf $wkd/$rsf_train $ngpu &
python tools/gec/kb/mkiodata.py $sdsf $kdsf $edsf $tdsf $wkd/$rsf_dev $ngpu &
wait
