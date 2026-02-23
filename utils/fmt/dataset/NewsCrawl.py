#encoding: utf-8

from os import walk
from os.path import join as pjoin

from utils.fmt.base import sys_open
from utils.fmt.fs.txt import line_reader as txt_line_reader
from utils.fmt.ifilter.b64 import ifilter as ifilter_b64
from utils.fmt.ifilter.decode import ifilter as ifilter_decode

def scan_train_set_sent(ptw):

	rs = []
	for root, dirs, files in walk(ptw):
		for f in files:
			if f.startswith("news.") and f.endswith(".gz"):
				rs.append(pjoin(root, f))

	return f

def scan_train_set_doc(ptw):

	rs = []
	for root, dirs, files in walk(ptw):
		for f in files:
			if f.startswith("news-docs.") and f.endswith(".gz"):
				rs.append(pjoin(root, f))

	return f

def doc_line_reader_core(fname):

	with sys_open(fname, "rb") as f:
		for _ in f:
			if _:
				_ = _.rstrip()
				if _:
					_ = _.split(b"\t")
					if len(_) == 3:
						_ = _[-1]
						if _:
							yield _

def doc_line_reader(fname, **kwargs):

	for _ in ifilter_decode(ifilter_b64(doc_line_reader_core(fname))):
		_ = _.lstrip()
		if _.rstrip():
			yield _

def get_line_readers(ptw, **kwargs):

	return [txt_line_reader(_) for _ in scan_train_set_sent(ptw)] + [doc_line_reader(_) for _ in scan_train_set_doc(ptw)]
