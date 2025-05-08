#encoding: utf-8

import sys
from transformers import Qwen2TokenizerFast as Tokenizer

from utils.fmt.base import iter_to_str, loop_file_so

from cnfg.vocab.plm.qwen.v3 import templated

def handle(fsrc, vcb, frs, template="instruct", system="You are a helpful assistant.", templated=templated, **kwargs):

	_tfunc = templated[template]
	_ = lambda lin, processor: " ".join(iter_to_str(processor(_tfunc(system, lin))))
	return loop_file_so(fsrc, frs, process_func=_, processor=Tokenizer.from_pretrained(vcb).encode)

if __name__ == "__main__":
	handle(*sys.argv[1:])
