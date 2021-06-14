import os
import sys
import pandas as pd
from globals import TOUR, START_YEAR, CURRENT_YEAR, RET_STRINGS, ABD_STRINGS, COUNTS_538
from data_functions import format_match_df, generate_df
from util import init_subdirectories, write_dataframe

sys.path.insert(0, '{}/sackmann'.format(os.path.dirname(os.path.abspath(__file__))))

# only need to run once, make sure to sort all matches in concat_data()
def format_match_data():
	print('formatting match data...')
	for file_name in os.listdir('../match_data'):
		df = pd.read_csv('../match_data/{}'.format(file_name))
		formatted_df = format_match_df(df, TOUR, RET_STRINGS, ABD_STRINGS)
		formatted_df.to_csv('../match_data_formatted/{}'.format(file_name), index=False)
	return

if __name__=='__main__':
	init_subdirectories()

	format_match_data()

	match_df = generate_df(TOUR, START_YEAR, CURRENT_YEAR, RET_STRINGS, ABD_STRINGS, COUNTS_538)

	write_dataframe(match_df, 'match')
