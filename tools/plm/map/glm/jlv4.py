#encoding: utf-8

import sys
from json import loads
from transformers import PreTrainedTokenizerFast as Tokenizer

from utils.fmt.base import iter_to_str, loop_file_so
from utils.fmt.plm.glm.base import add_prefix, lstrip

from cnfg.vocab.plm.glm.v4 import gmask_id, sop_id, templated

def handle(fsrc, vcb, frs, template="instruct", system="You are a helpful assistant.", templated=templated, lsl=[gmask_id, sop_id], **kwargs):

	_tfunc = templated[template]
	_ = (lambda lin, processor: " ".join(iter_to_str(lstrip(processor(_tfunc(system, loads(lin))), lsl)))) if template == "assistant" else (lambda lin, processor: " ".join(iter_to_str(add_prefix(processor(_tfunc(system, loads(lin))), lsl))))
	return loop_file_so(fsrc, frs, process_func=_, processor=Tokenizer.from_pretrained(vcb).encode)

if __name__ == "__main__":
	handle(*sys.argv[1:])
