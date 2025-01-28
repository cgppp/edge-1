#encoding: utf-8

import sys
from transformers import Qwen2TokenizerFast as Tokenizer

from utils.fmt.plm.token import map_back_file_jdumps as map_func

def handle(fsrc, vcb, frs):

	return map_func(fsrc, frs, processor=Tokenizer.from_pretrained(vcb).decode)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3])
