import os

from globals import *

def init_subdirectories():
	for subdir in ('match_data_constructed', 'match_data_formatted'):
		path_to_subdir = '../{}'.format(subdir)
		if not os.path.exists(path_to_subdir):
			os.mkdir(path_to_subdir)

def write_dataframe(df, prefix):
	file_path = '../match_data_constructed/{}_df_{}.csv'.format(prefix, DATE)
	df.to_csv(file_path, index=False)
	print('{} constructed'.format(file_path))