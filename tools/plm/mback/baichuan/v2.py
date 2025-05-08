#encoding: utf-8

import sys

from utils.fmt.plm.token import map_back_file_jdumps as map_func

def handle(fsrc, vcb, frs, **kwargs):

	sys.path.append(vcb)
	from tokenization_baichuan import BaichuanTokenizer as Tokenizer

	return map_func(fsrc, frs, processor=Tokenizer.from_pretrained(vcb).decode)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3])
