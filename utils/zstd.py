#encoding: utf-8

from io import BufferedReader
from zstandard import ZstdDecompressor

def line_reader(fname):

	with open(fname, "rb") as f, ZstdDecompressor().stream_reader(f) as s, BufferedReader(s) as b:
		for _ in b:
			yield _
