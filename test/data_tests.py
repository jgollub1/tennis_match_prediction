import pandas as pd
import numpy as np
from globals import COUNTS_538, TOUR, RET_STRINGS, ABD_STRINGS
from data_functions import *
from tqdm import tqdm

START_YEAR_TEST = 2012
END_YEAR_TEST = 2015
CURRENT_DF_TEST_PATH = 'test_data/test_df_current.csv'
MATCH_DF_TEST_PATH = 'test_data/test_df_match.csv'
TEST_COLUMNS_ELO = ['w_elo_538', 'l_elo_538', 'w_sf_elo_538', 'l_sf_elo_538']
TEST_COLUMNS_52 = [
    'w_52_swon', 'w_52_svpt', 'w_52_rwon', 'w_52_rpt',
    'w_sf_52_swon', 'w_sf_52_svpt', 'w_sf_52_rwon', 'w_sf_52_rpt',
    'l_52_swon', 'l_52_svpt', 'l_52_rwon', 'l_52_rpt',
    'l_sf_52_swon', 'l_sf_52_svpt', 'l_sf_52_rwon', 'l_sf_52_rpt',
    'avg_52_s', 'avg_52_r'
]
TEST_COLUMNS_52_ADJ = [
    'w_52_s_adj', 'w_52_r_adj', 'l_52_s_adj', 'l_52_r_adj'
]
TEST_COLUMNS_COMMOP = [
    'w_commop_s_pct', 'w_commop_r_pct', 'l_commop_s_pct', 'l_commop_r_pct'
]

def compare_cols(df, test_df, col_name):
    try:
        assert(np.isclose(df[col_name], test_df[col_name]).all())
    except:
        print 'failed at col: ', col_name

def test_cols(df, test_df, cols):
    for col in tqdm(cols):
        compare_cols(df, test_df, col)

# TODO: update tests to use match_df not active df
def test_elo(df, test_df):
    print '### testing elo ###'
    elo_df = generate_elo(df, COUNTS_538)
    print 'generated: ', elo_df.shape
    test_cols(elo_df, test_df, TEST_COLUMNS_ELO)
    print '--- elo passed ---'

def test_52_stats(df, test_df):
    print '### testing 52 stats ###'
    df = generate_52_stats(df, 0)
    test_cols(df, test_df, TEST_COLUMNS_52)
    print '--- 52 stats passed ---'

def test_52_adj_stats(df, test_df):
    print '### testing 52 adj stats ###'
    df = generate_52_adj_stats(df, 0)
    test_cols(df, test_df, TEST_COLUMNS_52_ADJ)
    print '--- 52 adj stats passed ---'

def test_commop_stats(df, test_df):
    print '### testing commop stats ###'
    df = generate_commop_stats(df, 0)
    test_cols(df, test_df, TEST_COLUMNS_COMMOP)
    assert(np.isclose(df['w_commop_s_pct'] + df['l_commop_s_pct'], 1.2).all())
    assert(np.isclose(df['w_commop_r_pct'] + df['l_commop_r_pct'], .8).all())
    print '--- commop stats passed ---'

# def test_em_stats(df, test_df):
#     print '### testing commop stats ###'
#     df = generate_(df, 0)
#     test_cols(df, test_df, TEST_COLUMNS_COMMOP)
#     print '--- EM stats passed ---'

def validate_data_pipeline(df, test_df):
    test_elo(df, test_df)

    test_52_stats(df, test_df)

    test_52_adj_stats(df, test_df)

    test_commop_stats(df, test_df)

def validate_test_df(test_df):
    return

if __name__=='__main__':
    df = concat_data(START_YEAR_TEST, END_YEAR_TEST, TOUR)

    test_df = pd.read_csv(MATCH_DF_TEST_PATH)

    validate_test_df(test_df)
    validate_data_pipeline(df, test_df)


# # only run this once
# def build_test_df():
#     match_df = generate_test_dfs(TOUR, 2012, 2015, RET_STRINGS, ABD_STRINGS, COUNTS_538)

#     match_file_path = 'test_data/test_df_match.csv'
#     match_df.to_csv(match_file_path, index=False)
#     print '{} constructed'.format(match_file_path)