#encoding: utf-8

from io import BufferedReader
from zstandard import ZstdDecompressor

from utils.fmt.base import sys_open

def line_reader(fname, **kwargs):

	with sys_open(fname, "rb") as f, ZstdDecompressor().stream_reader(f) as s, BufferedReader(s) as b:
		for _ in b:
			yield _
