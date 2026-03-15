#encoding: utf-8
"""
将「按行存储的文本文件」按指定 Qwen 指令模板格式化后，用 Qwen tokenizer 转成 token id，写出为「每行空格分隔的 id」文件。

用法（命令行）:
  python tools/plm/map/qwen/v3.py <输入文本路径> <tokenizer 路径> <输出 .ids 路径> [模板名] [system 提示]
例如:
  python tools/plm/map/qwen/v3.py $WKD/src.train.txt $TOKENIZER $WKD/src.train.txt.ids instruct_auto
  python tools/plm/map/qwen/v3.py $WKD/tgt.train.txt $TOKENIZER $WKD/tgt.train.txt.ids assistant

与 Qwen3 兼容：Qwen2TokenizerFast 与 Qwen3 同词表，模板来自 cnfg.vocab.plm.qwen.v3。
"""

import sys
from transformers import Qwen2TokenizerFast as Tokenizer

from utils.fmt.base import iter_to_str, loop_file_so

from cnfg.vocab.plm.qwen.v3 import templated


def handle(fsrc, vcb, frs, template="instruct", system="You are a helpful assistant.", templated=templated, **kwargs):
	"""
	逐行读入文本，按模板 + system 拼成字符串，tokenize 后写出为每行空格分隔的 id。

	参数:
		fsrc (str): 输入文本文件路径，每行一条样本（如一条指令或一条回答）。
		vcb (str): Qwen tokenizer 所在目录（如 /path/to/Qwen3-0.6B），用于 Tokenizer.from_pretrained(vcb)。
		frs (str): 输出文件路径，每行为该行文本经模板与 tokenize 后的 id，空格分隔（即 .ids 文件）。
		template (str): 模板名，取值于 cnfg.vocab.plm.qwen.v3.templated，如 "instruct_auto"（指令侧）、"assistant"（回答侧）。
		system (str): 填入模板的「系统提示」；部分模板无 system 则忽略。
		templated (dict): 模板名 -> 模板函数，默认使用 cnfg.vocab.plm.qwen.v3.templated。
		**kwargs: 保留，未使用。

	返回:
		loop_file_so 的返回值（通常为 None 或写文件结果）。
	"""
	_tfunc = templated[template]   # 根据模板名取出模板函数，如 instruct_auto -> instruct_auto_template
	# 单行处理：lin 为当前行文本，processor 为 tokenizer.encode；先用模板拼成字符串，再 encode，再转成空格分隔的 id 串
	_ = lambda lin, processor: " ".join(iter_to_str(processor(_tfunc(system, lin))))
	# 逐行读 fsrc，对每行应用 _，用 Qwen tokenizer 编码，结果写入 frs
	return loop_file_so(fsrc, frs, process_func=_, processor=Tokenizer.from_pretrained(vcb).encode)


if __name__ == "__main__":
	# 命令行：python v3.py <fsrc> <vcb> <frs> [template] [system]
	handle(*sys.argv[1:])
