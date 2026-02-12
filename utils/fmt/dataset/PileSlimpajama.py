#encoding: utf-8

from os import walk
from os.path import join as pjoin

from utils.fmt.fs.parquet import line_reader as pq_line_reader
from utils.fmt.ifilter.emp import ifilter

def scan_train_set(ptw):

	rs = []
	for root, dirs, files in walk(ptw):
		for f in files:
			if f.startswith("train-") and f.endswith(".parquet"):
				rs.append(pjoin(root, f))

	return f

def line_reader_core(fname):

	for _ in pq_line_reader(fname):
		yield _.get("text", "")

def line_reader(fname, **kwargs):

	return ifilter(line_reader_core(fname))

def get_line_readers(ptw, **kwargs):

	return [line_reader(_) for _ in scan_train_set(ptw)]
