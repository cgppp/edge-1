#encoding: utf-8

import sys
from json import loads
from transformers import PreTrainedTokenizerFast as Tokenizer

from utils.fmt.base import loop_file_so

from cnfg.vocab.plm.llama.v3 import templated

def handle(fsrc, vcb, frs, template="instruct", system="You are a helpful assistant.", templated=templated):

	_tfunc = templated[template]
	_ = lambda lin, processor: " ".join(processor(_tfunc(system, loads(lin))))
	return loop_file_so(fsrc, frs, process_func=_, processor=Tokenizer.from_pretrained(vcb).tokenize)

if __name__ == "__main__":
	handle(*sys.argv[1:])
