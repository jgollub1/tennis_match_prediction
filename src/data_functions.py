import os
import sys

sys.path.insert(0, './sackmann')

import re
import datetime
import numpy as np
import pandas as pd
import elo_538 as elo
from tennisMatchProbability import matchProb
from processing_util import normalize_name
from data_classes import stats_52, adj_stats_52, tny_52, commop_stats
from collections import defaultdict
from globals import COMMOP_START_YEAR, EPSILON

pd.options.mode.chained_assignment = None

'''
concatenate original match dataframes from years
(start_y, end_y)
'''
def concat_data(start_y, end_y, tour):
    match_year_list = []
    for i in range(start_y, end_y+1):
        f_name = "../match_data_formatted/{}_matches_{}.csv".format(tour, i)
        try:
            match_year_list.append(pd.read_csv(f_name))
        except:
            print('could not find file for year: ', i)
    full_match_df = pd.concat(match_year_list, ignore_index = True)
    return full_match_df.sort_values(by=['tny_date','tny_name','match_num'], ascending=True).reset_index(drop=True)

'''
match data preprocessing
'''
def format_match_df(df,tour,ret_strings=[],abd_strings=[]):
    cols = [u'tourney_id', u'tourney_name', u'surface', u'draw_size', u'tourney_date',
            u'match_num', u'winner_name', u'loser_name', u'score', u'best_of', u'w_svpt',
            u'w_1stWon', u'w_2ndWon', u'l_svpt', u'l_1stWon', u'l_2ndWon']
    df = df[cols]
    df = df.rename(columns={'winner_name':'w_name','loser_name':'l_name','tourney_id':'tny_id',\
                            'tourney_name':'tny_name','tourney_date':'tny_date'})

    df['w_name'] = [normalize_name(x,tour) for x in df['w_name']]
    df['l_name'] = [normalize_name(x,tour) for x in df['l_name']]
    df['tny_name'] = ['Davis Cup' if 'Davis Cup' in s else s for s in df['tny_name']]
    df['tny_name'] = [s.replace('Australian Chps.','Australian Open').replace('Australian Open-2',\
                'Australian Open').replace('U.S. National Chps.','US Open') for s in df['tny_name']]
    df['is_gs'] = (df['tny_name'] == 'Australian Open') | (df['tny_name'] == 'Roland Garros') |\
                  (df['tny_name'] == 'Wimbledon')       | (df['tny_name'] == 'US Open')

    # format dates
    df['tny_date'] = [datetime.datetime.strptime(str(x), "%Y%m%d").date() for x in df['tny_date']]
    df['match_year'] = [x.year for x in df['tny_date']]
    df['match_month'] = [x.month for x in df['tny_date']]
    df['match_year'] = df['match_year'] + (df['match_month'] == 12) # correct december start dates
    df['match_month'] = [1 if month==12 else month for month in df['match_month']] # to following year
    df['score'] = [re.sub(r"[\(\[].*?[\)\]]", "", str(s)) for s in df['score']] # str(s) fixes any nans
    df['score'] = ['RET' if 'RET' in s else s for s in df['score']]
    df['w_swon'], df['l_swon'] = df['w_1stWon']+df['w_2ndWon'], df['l_1stWon']+df['l_2ndWon']
    df['w_rwon'], df['l_rwon'] = df['l_svpt']-df['l_swon'], df['w_svpt']-df['w_swon']
    df['w_rpt'], df['l_rpt'] = df['l_svpt'], df['w_svpt']
    df.drop(['w_1stWon','w_2ndWon','l_1stWon','l_2ndWon'], axis=1, inplace=True)

    # remove matches involving a retirement
    abd_d, ret_d = set(abd_strings), set(ret_strings)
    df['score'] = ['ABN' if score.split(' ')[-1] in abd_d else score for score in df['score']]
    df['score'] = ['RET' if score in ret_d else score for score in df['score']]
    return df.loc[(df['score'] != 'ABN') & (df['score'] != 'RET')].reset_index(drop=True)

'''
original dataset labels columns by 'w_'/'l_'
change 'w'/'l' to 'p0','p1' (where p0 is the higher ranked player, according to Elo ratings)
'''
# TODO: refactor this into two functions
def change_labels(df, cols):
    # change w,l TO p0,p1
    for col in cols:
        df['p0'+col] = [df['l'+col][i] if df['winner'][i] else df['w'+col][i] for i in range(len(df))]
        df['p1'+col] = [df['w'+col][i] if df['winner'][i] else df['l'+col][i] for i in range(len(df))]

    # add s/r pct columns
    p_hat = np.sum([df['p0_52_swon'],df['p1_52_swon']])/np.sum([df['p0_52_svpt'],df['p1_52_svpt']])
    for label in ['p0','p1']:
        df[label+'_s_pct'] = [p_hat if x==0 else x for x in np.nan_to_num(df[label+'_52_swon']/df[label+'_52_svpt'])]
        df[label+'_r_pct'] = [1-p_hat if x==0 else x for x in np.nan_to_num(df[label+'_52_rwon']/df[label+'_52_rpt'])]
        df[label+'_sf_s_pct'] = [p_hat if x==0 else x for x in np.nan_to_num(df[label+'_sf_52_swon']/df[label+'_sf_52_svpt'])]
        df[label+'_sf_r_pct'] = [1-p_hat if x==0 else x for x in np.nan_to_num(df[label+'_sf_52_rwon']/df[label+'_sf_52_rpt'])]

    for label in ['w', 'l']:
        df.drop([label + col for col in cols], axis=1, inplace=True)

    df['tny_name'] = [s if s==s else 'Davis Cup' for s in df['tny_name']]
    return df

'''
original dataset labels columns by 'w_'/'l_'
change 'w'/'l' to 'p0','p1' (where p0 is the higher ranked player, according to Elo ratings)
(without extra formatting)
'''
def change_labels_v2(df, cols):
    # change w,l TO p0,p1
    for col in cols:
        df['p0'+col] = [df['l'+col][i] if df['winner'][i] else df['w'+col][i] for i in range(len(df))]
        df['p1'+col] = [df['w'+col][i] if df['winner'][i] else df['l'+col][i] for i in range(len(df))]

    for label in ['w', 'l']:
        df.drop([label + col for col in cols], axis=1, inplace=True)

    return df

'''
confirm that match serve/return stats are not null
'''
def validate(row, label):
    return row[label+'_swon']==row[label+'_swon'] and row[label+'_svpt']==row[label+'_svpt'] \
        and row[label+'_rwon']==row[label+'_rwon'] and row[label+'_rpt']==row[label+'_rpt']

'''
from start_ind (a year before start_year), collect cumulative
12-month s/r stats prior to each match
'''
def get_current_52_stats(df, start_ind):
    players_stats = {}
    active_players = {}
    w_l = ['p0', 'p1']
    start_date = (df['match_year'][start_ind],df['match_month'][start_ind])
    avg_stats = stats_52(start_date)
    avg_stats.update(start_date,(6.4,10,3.6,10)) # set as prior so first row is not nan

    for i, row in df[start_ind:].iterrows():
        date = row['match_year'],row['match_month']
        avg_stats.set_month(date)
        for k,label in enumerate(w_l):
            if row[label+'_name'] not in players_stats:
                players_stats[row[label+'_name']] = stats_52(date)
            # store serving stats prior to match, update current month
            players_stats[row[label+'_name']].set_month(date)
            if validate(row, label):
                match_stats = (row[label+'_swon'],row[label+'_svpt'],row[w_l[1-k]+'_svpt']-\
                                row[w_l[1-k]+'_swon'],row[w_l[1-k]+'_svpt'])
                players_stats[row[label+'_name']].update(date,match_stats)
                avg_stats.update(date,match_stats)

            active_players[row[label+'_name']] = 1 # log active player

    # update every player to current month
    for player in active_players.keys():
        players_stats[player].set_month(date)

    players = active_players.keys()
    current_52_stats = [[player] + list(np.sum(players_stats[player].last_year,axis=0)) \
                        for player in players]
    # avg_52_stats = np.sum(avg_stats.last_year,axis=0)
    cols = ['player','52_swon','52_svpt','52_rwon','52_rpt']
    current_stats_df = pd.DataFrame(current_52_stats, columns=cols)
    current_stats_df['52_s_pct'] = current_stats_df['52_swon']/current_stats_df['52_svpt']
    current_stats_df['52_r_pct'] = current_stats_df['52_rwon']/current_stats_df['52_rpt']
    return current_stats_df[current_stats_df['52_svpt']>0] # return players active in past 12 months

'''
generate 12-month stats for Barnett-Clarke model
as well as variations (adjusted, EM-normalized)
'''
def generate_stats(df, start_ind):
    df = generate_52_stats(df,start_ind)
    print('generated 52 stats...')

    df = generate_52_adj_stats(df,start_ind)
    print('generated 52 adj stats...')

    df = generate_tny_stats(df,start_ind)
    print('generated tny stats...')

    df = generate_commop_stats(df, start_ind)
    print('generated commop stats...')

    cols = ['_name','_elo_538','_sf_elo_538', #'_elo','_sf_elo'
        '_swon', '_svpt', '_rwon', '_rpt',
        '_52_swon', '_52_svpt','_52_rwon','_52_rpt',
        '_sf_52_swon','_sf_52_svpt','_sf_52_rwon','_sf_52_rpt',
        '_52_s_adj','_52_r_adj']

    # of players p0, p1, p0 will always be the player with the first name alphabetically (since this is deterministic)
    # the 'winner' will be 1 when p0's name comes alphabetically last and 0 otherwise
    df['winner'] = df['w_name'] > df['l_name']
    df = change_labels(df, cols)
    df = change_labels_v2(df, ['_commop_s_pct', '_commop_r_pct'])

    df['elo_diff'] = df['p0_elo_538'] - df['p1_elo_538']
    df['sf_elo_diff'] = df['p0_sf_elo_538'] - df['p1_sf_elo_538']

    # # dataframe with only official matches
    # df = df[df['winner']!='None']
    # df = df.reset_index(drop=True)
    # cols = ['52_s_adj','52_r_adj']

    em_cols = ['s_pct', 'r_pct', 'sf_s_pct', 'sf_r_pct', '52_s_adj', '52_r_adj']
    df = generate_sr_pct(df)

    # FIX for correct em stat sample sizes
    df = df.loc[start_ind:].reset_index(drop=True)
    df = generate_em_stats(df, em_cols)
    return df

'''
add s/r pct columns, replacing nan with overall avg
'''
def generate_sr_pct(df):
    p_hat = np.sum([df['p0_52_swon'],df['p1_52_swon']])
    p_hat = p_hat/np.sum([df['p0_52_svpt'],df['p1_52_svpt']])
    for label in ['p0','p1']:
        # divide with np.nan_to_num and use p_hat as a placeholder when n=0
        df[label+'_s_pct'] = np.nan_to_num(df[label+'_52_swon']/df[label+'_52_svpt'])
        df[label+'_s_pct'] = df[label+'_s_pct'] + (p_hat) * (df[label+'_s_pct'] == 0)
        df[label+'_r_pct'] = np.nan_to_num(df[label+'_52_rwon']/df[label+'_52_rpt'])
        df[label+'_r_pct'] = df[label+'_r_pct'] + (1-p_hat)*(df[label+'_r_pct'] == 0)

        df[label+'_sf_s_pct'] = np.nan_to_num(df[label+'_sf_52_swon']/df[label+'_sf_52_svpt'])
        df[label+'_sf_s_pct'] = df[label+'_sf_s_pct'] + (p_hat) * (df[label+'_sf_s_pct'] == 0)
        df[label+'_sf_r_pct'] = np.nan_to_num(df[label+'_sf_52_rwon']/df[label+'_sf_52_rpt'])
        df[label+'_sf_r_pct'] = df[label+'_sf_r_pct'] + (1-p_hat)*(df[label+'_sf_r_pct'] == 0)

        # finally, generate the observed service percentages in each match
        df[label+'_s_pct_obsv'] = np.nan_to_num(df[label+'_swon']/df[label+'_svpt'])
    return df

def finalize_df(df):
    # generate serving probabilities for Barnett-Clarke model
    df['match_id'] = range(len(df))
    df['tny_stats'] = [df['avg_52_s'][i] if df['tny_stats'][i]==0 else df['tny_stats'][i] for i in range(len(df))]
    df['p0_s_kls'] = df['tny_stats']+(df['p0_s_pct']-df['avg_52_s']) - (df['p1_r_pct']-df['avg_52_r'])
    df['p1_s_kls'] = df['tny_stats']+(df['p1_s_pct']-df['avg_52_s']) - (df['p0_r_pct']-df['avg_52_r'])
    df['p0_s_kls_EM'] = df['tny_stats']+(df['p0_s_pct_EM']-df['avg_52_s']) - (df['p1_r_pct_EM']-df['avg_52_r'])
    df['p1_s_kls_EM'] = df['tny_stats']+(df['p1_s_pct_EM']-df['avg_52_s']) - (df['p0_r_pct_EM']-df['avg_52_r'])

    df['p0_s_sf_kls'] = df['tny_stats']+(df['p0_sf_s_pct']-df['sf_avg_52_s']) - (df['p1_sf_r_pct']-df['sf_avg_52_r'])
    df['p1_s_sf_kls'] = df['tny_stats']+(df['p1_sf_s_pct']-df['sf_avg_52_s']) - (df['p0_sf_r_pct']-df['sf_avg_52_r'])
    df['p0_s_sf_kls_EM'] = df['tny_stats']+(df['p0_sf_s_pct_EM']-df['sf_avg_52_s']) - (df['p1_sf_r_pct_EM']-df['sf_avg_52_r'])
    df['p1_s_sf_kls_EM'] = df['tny_stats']+(df['p1_sf_s_pct_EM']-df['sf_avg_52_s']) - (df['p0_sf_r_pct_EM']-df['sf_avg_52_r'])

    df['p0_s_adj_kls'] = df['tny_stats']+(df['p0_52_s_adj']) - (df['p1_52_r_adj'])
    df['p1_s_adj_kls'] = df['tny_stats']+(df['p1_52_s_adj']) - (df['p0_52_r_adj'])
    df['p0_s_adj_kls_EM'] = df['tny_stats']+(df['p0_52_s_adj_EM']) - (df['p1_52_r_adj_EM'])
    df['p1_s_adj_kls_EM'] = df['tny_stats']+(df['p1_52_s_adj_EM']) - (df['p0_52_r_adj_EM'])

    df['p0_s_commop_kls'] = df['tny_stats']+(df['p0_commop_s_pct'] - df['avg_52_s']) - (df['p1_commop_r_pct'] - df['avg_52_r'])
    df['p1_s_commop_kls'] = df['tny_stats']+(df['p1_commop_s_pct'] - df['avg_52_s']) - (df['p0_commop_r_pct'] - df['avg_52_r'])

    p_hat = np.sum([df['p0_52_swon'],df['p1_52_swon']])/np.sum([df['p0_52_svpt'],df['p1_52_svpt']])
    df['p0_s_baseline'] = p_hat
    df['p1_s_baseline'] = p_hat

    # generate match probabilities for Barnett-Clarke method, with or w/o EM estimators
    df['match_prob_kls'] = [matchProb(row['p0_s_kls'],1-row['p1_s_kls']) for i,row in df.iterrows()]
    df['match_prob_kls_EM'] = [matchProb(row['p0_s_kls_EM'],1-row['p1_s_kls_EM']) for i,row in df.iterrows()]
    df['match_prob_sf_kls'] = [matchProb(row['p0_s_sf_kls'],1-row['p1_s_sf_kls']) for i,row in df.iterrows()]
    df['match_prob_sf_kls_EM'] = [matchProb(row['p0_s_sf_kls_EM'],1-row['p1_s_sf_kls_EM']) for i,row in df.iterrows()]
    df['match_prob_adj_kls'] = [matchProb(row['p0_s_adj_kls'],1-row['p1_s_adj_kls']) for i,row in df.iterrows()]
    df['match_prob_adj_kls_EM'] = [matchProb(row['p0_s_adj_kls_EM'],1-row['p1_s_adj_kls_EM']) for i,row in df.iterrows()]
    df['match_prob_commop_kls'] = [matchProb(row['p0_s_commop_kls'],1-row['p1_s_commop_kls']) for i,row in df.iterrows()]
    df['match_prob_commop'] = [1 - df['w_commop_match_prob'][i] if df['winner'][i] else df['w_commop_match_prob'][i] for i in range(len(df))]

    # generate win probabilities from elo differences
    df['elo_prob'] = (1+10**(df['elo_diff']/-400.))**-1
    df['sf_elo_prob'] = [(1+10**(diff/-400.))**-1 for diff in df['sf_elo_diff']]

    # elo-induced serve percentages
    df = generate_bc_stats_elo_induced(df, 'elo',start_ind=0)
    return df

def get_start_ind(match_df, start_year):
    return match_df[match_df['match_year']>=start_year-1].index[0]

'''
returns dataframe with up-to-date player stats through date of most recent match
'''
def generate_df(tour, start_year, end_year, ret_strings, abd_strings, counts_538):
    print('start_year: ', start_year)
    print('end_year: ', end_year)
    match_df = concat_data(start_year, end_year, tour)
    print('match_df.shape before: ', match_df.shape)
    start_ind = match_df[match_df['match_year']>=start_year-1].index[0]
    print('match_df.shape: ', match_df.shape)
    match_df = generate_elo(match_df, counts_538)
    print('generated elo on match dataset...')

    match_df = generate_stats(match_df, start_ind) # 52, adj, tny, etc.
    match_df = finalize_df(match_df)
    match_df = match_df.reset_index(drop=True)
    print('finalized df...')
    return match_df

'''
returns two dataframes
1) contains up-to-date player stats through date of most recent match
2) contains every match with elo/serve/return/etc stats
'''
def generate_test_dfs(tour, start_year, end_year, ret_strings, abd_strings, counts_538):
    match_df = concat_data(start_year, end_year, tour)
    start_ind = match_df[match_df['match_year']>=start_year-1].index[0]
    match_df = generate_elo(match_df, counts_538)

    match_df = generate_52_stats(match_df, start_ind)
    match_df = generate_52_adj_stats(match_df, start_ind)
    match_df = generate_tny_stats(match_df, start_ind)
    match_df = generate_commop_stats(match_df, start_ind)
    # TODO: add generate_em_stats() right here

    return match_df

'''
receives n x 2 array with columns 'w_name', 'l_name', 'is_gs'
'''
def generate_elo_columns(arr, counts_538):
    # print('arr[0]: ', arr[0])
    # print('arr[1]: ', arr[1])
    # print('arr[0].dtype', arr[:, 0].dtype)
    # print('arr[1].dtype', arr[:, 1].dtype)
    # print('arr[:, :2]', arr[:, :2])
    # for s in arr:
    #     if isinstance(s[1], bool):
    #         print('s: ', s)
    # print('is: ', [s for s in arr if isinstance(s[0], bool))
    # players_set = np.unique(arr[:, :2].astype(str))
    player_names = arr[:, :2].flatten()
    players_set = np.where(player_names!=player_names, '', player_names).tolist()
    # players_set = list(set(list(np.concatenate(arr[:, 0], arr[:, 1]))))
    # player_count = len(players_set)
    # print('player_count: ', player_count)
    # initial_ratings = [elo.Rating() for _ in range(player_count)]
    # zipped = zip(
    #     players_set,
    #     [elo.Rating() for _ in range(player_count)]
    # )
    # # print('zipped: ', zipped)
    # players_elo = dict(zip(
    #     players_set,
    #     [elo.Rating() for _ in range(player_count)]
    # )) # can use default dict here?
    players_elo = {}
    for player in players_set:
        # print('player: ', player)
        players_elo[player] = elo.Rating()

    match_elos = np.zeros([arr.shape[0], 2])
    elo_obj = elo.Elo_Rater()

    # update player elo from every recorded match
    for i in range(arr.shape[0]):
        w_name, l_name = arr[i][:2]

        if w_name != w_name or l_name != l_name:
            match_elos[i] = np.nan, np.nan
            continue

        match_elos[i] = players_elo[w_name].value, players_elo[l_name].value
        elo_obj.rate_1vs1(players_elo[w_name], players_elo[l_name], arr[i][2], counts_538)

    return match_elos[:,0], match_elos[:,1]

def generate_surface_elo_columns(df, surfaces, counts_538):
    df['w_sf_elo_538'], df['l_sf_elo_538'] = df['w_elo_538'], df['l_elo_538']
    for surface in surfaces:
        surface_df = df[(df['surface'] == surface) & (df['w_name'] == df['w_name']) & (df['l_name'] == df['l_name'])]
        w_elo_columns, l_elo_columns = generate_elo_columns(np.array(surface_df[['w_name', 'l_name', 'is_gs']]), True)
        df.loc[df['surface'] == surface, 'w_sf_elo_538'] = w_elo_columns
        df.loc[df['surface'] == surface, 'l_sf_elo_538'] = l_elo_columns

    return df['w_sf_elo_538'], df['l_sf_elo_538']

'''
receives n x 4 array with columns 'w_name', 'l_name', 'is_gs', 'Date'
'''
def generateEloColumnsWithHistory(arr, counts_538):
    playerEloHistory = defaultdict(list)
    players_set = np.unique(arr[:, :2])
    players_elo = dict(zip(
        players_set,
        [elo.Rating() for __ in range(len(players_set))]
    )) # can use default dict here?

    match_elos = np.zeros([arr.shape[0], 2])
    elo_obj = elo.Elo_Rater()

    # update player elo from every recorded match
    for i in range(arr.shape[0]):
        w_name, l_name = arr[i][:2]
        isGrandSlam = arr[i][2]
        date = datetime.datetime.strptime(arr[i][3], '%Y-%m-%d')

        match_elos[i] = players_elo[w_name].value, players_elo[l_name].value
        elo_obj.rate_1vs1(players_elo[w_name], players_elo[l_name], 0, counts_538)

        playerEloHistory[w_name].append({ 'date': date, 'newElo': players_elo[w_name].value, 'won': 1 })
        playerEloHistory[l_name].append({ 'date': date, 'newElo': players_elo[l_name].value, 'won': 0 })

    return match_elos[:,0], match_elos[:,1], playerEloHistory, players_elo

'''
return match dataframe with each player's pre-match elo ratings
'''
def generate_elo(df, counts_538=True):
    df['w_elo_538'], df['l_elo_538'] = generate_elo_columns(np.array(df[['w_name', 'l_name', 'is_gs']]), True)
    df['w_sf_elo_538'], df['l_sf_elo_538'] = generate_surface_elo_columns(df, ['Hard', 'Clay', 'Grass'], counts_538)
    return df

    # df['w_sf_elo_538'], df['l_sf_elo_538'] = df['w_elo_538'], df['l_elo_538']
    # for surface in ['Hard', 'Clay', 'Grass']:
    #     surface_df = df[df['surface'] == surface]
    #     w_elo_columns, l_elo_columns = generate_elo_columns(np.array(surface_df[['w_name', 'l_name', 'is_gs']]), True)
    #     df.loc[df['surface'] == surface, 'w_sf_elo_538'] = w_elo_columns
    #     df.loc[df['surface'] == surface, 'l_sf_elo_538'] = l_elo_columns

    # return df

'''
replace nan values with overall average array value
'''
def fill_nan_with_mean(arr):
    mean = np.nanmean(arr)
    arr[np.isnan(arr)] = mean
    return arr

'''
collect 12-month s/r average performance by player
'''
def generate_52_stats(df,start_ind):
    players_stats = {}
    start_date = (df['match_year'][start_ind],df['match_month'][start_ind])
    avg_stats = stats_52(start_date)
    # set as prior so first row is not nan
    avg_stats.update(start_date,(6.4,10,3.6,10))
    # array w/ 2x1 arrays for each player's 12-month serve/return performance
    match_52_stats = np.zeros([2,len(df),4])
    avg_52_stats = np.zeros([len(df),4]) # avg tour-wide stats for serve, return

    s_players_stats = {}
    s_avg_stats = {}
    for surface in ('Hard','Clay','Grass'):
        s_players_stats[surface] = {}
        s_avg_stats[surface] = stats_52((df['match_year'][0],df['match_month'][0]))
        s_avg_stats[surface].update(start_date,(6.4,10,3.6,10))
    s_match_52_stats = np.zeros([2,len(df),4])
    s_avg_52_stats = np.zeros([len(df),4])

    w_l = ['w','l']
    for i, row in df.loc[start_ind:].iterrows():
        surface = row['surface']
        date = row['match_year'],row['match_month']

        avg_stats.set_month(date)
        avg_52_stats[i] = np.sum(avg_stats.last_year,axis=0)
        for k,label in enumerate(w_l):
            if row[label+'_name'] not in players_stats:
                players_stats[row[label+'_name']] = stats_52(date)
            # store serving stats prior to match, update current month
            players_stats[row[label+'_name']].set_month(date)
            match_52_stats[k][i] = np.sum(players_stats[row[label+'_name']].last_year,axis=0) # all four stats per player
            # update serving stats if not null
            if validate(row, label):
                match_stats = (row[label+'_swon'],row[label+'_svpt'],row[w_l[1-k]+'_svpt']-\
                                row[w_l[1-k]+'_swon'],row[w_l[1-k]+'_svpt'])
                players_stats[row[label+'_name']].update(date,match_stats)
                avg_stats.update(date,match_stats)

        # repeat above process for surface-specific stats
        if surface not in ('Hard','Clay','Grass'):
            continue
        s_avg_stats[surface].set_month(date)
        s_avg_52_stats[i] = np.sum(s_avg_stats[surface].last_year,axis=0)
        for k,label in enumerate(w_l):
            if row[label+'_name'] not in s_players_stats[surface]:
                s_players_stats[surface][row[label+'_name']] = stats_52(date)

            # store serving stats prior to match, from current month
            s_players_stats[surface][row[label+'_name']].set_month(date)
            s_match_52_stats[k][i] = np.sum(s_players_stats[surface][row[label+'_name']].last_year,axis=0)
            # update serving stats if not null
            if validate(row, label):
                match_stats = (row[label+'_swon'],row[label+'_svpt'],row[w_l[1-k]+'_svpt']-\
                                row[w_l[1-k]+'_swon'],row[w_l[1-k]+'_svpt'])
                s_players_stats[surface][row[label+'_name']].update(date,match_stats)
                s_avg_stats[surface].update(date,match_stats)

    for k,label in enumerate(w_l):
        df[label+'_52_swon'] = match_52_stats[k][:,0]
        df[label+'_52_svpt'] = match_52_stats[k][:,1]
        df[label+'_52_rwon'] = match_52_stats[k][:,2]
        df[label+'_52_rpt'] = match_52_stats[k][:,3]
        df[label+'_sf_52_swon'] = s_match_52_stats[k][:,0]
        df[label+'_sf_52_svpt'] = s_match_52_stats[k][:,1]
        df[label+'_sf_52_rwon'] = s_match_52_stats[k][:,2]
        df[label+'_sf_52_rpt'] = s_match_52_stats[k][:,3]

    with np.errstate(divide='ignore', invalid='ignore'):
        df['avg_52_s'] = fill_nan_with_mean(np.divide(avg_52_stats[:,0],avg_52_stats[:,1]))
        df['avg_52_r'] = fill_nan_with_mean(np.divide(avg_52_stats[:,2],avg_52_stats[:,3]))
        df['sf_avg_52_s'] = fill_nan_with_mean(np.divide(s_avg_52_stats[:,0],s_avg_52_stats[:,1]))
        df['sf_avg_52_r'] = fill_nan_with_mean(np.divide(s_avg_52_stats[:,2],s_avg_52_stats[:,3]))
    return df

'''
Efron-Morris estimators for 52-week serve and return percentages
Calculates B_i coefficients in terms of service points
Feed any existing col where ['p0_'+col, 'p1_'+col] within df.columns
# TODO: you should be passing in the full column suffix after 'p0_'/'p1_'
'''
def generate_em_stats(df,cols):
    for col in cols:
        stat_history = np.concatenate([df['p0_'+col],df['p1_'+col]],axis=0)
        n = int(len(stat_history)/2)
        prefix = 'sf_52_' if 'sf' in col else '52_'
        suffix = 'svpt' if '_s_' in col else 'rpt'
        num_points = np.concatenate([df['p0_'+prefix+suffix],df['p1_'+prefix+suffix]])
        p_hat = np.mean(stat_history)
        sigma2_i = fill_nan_with_mean(np.divide(p_hat*(1-p_hat),num_points,where=num_points>0))
        tau2_hat = np.nanvar(stat_history)
        B_i = sigma2_i/(tau2_hat+sigma2_i)

        stat_history[stat_history!=stat_history] = p_hat
        df['p0_' + col + '_EM'] = df['p0_' + col]+B_i[:n] * (p_hat - df['p0_' + col])
        df['p1_' + col + '_EM'] = df['p1_' + col]+B_i[n:] * (p_hat - df['p1_' + col])
        print(col, p_hat)
    return df # ok if p_hats don't add up because they're avg of averages

'''
Efron-Morris estimators for 52-week serve and return percentages
Calculates B_i coefficients in terms of service points
Feed any existing col within df.columns
'''
def generate_em_stats_current(df,cols):
    for col in cols:
        stat_history = df[col]
        num_points = df['52_svpt'] if col=='52_swon' else df['52_rpt']
        p_hat = np.mean(stat_history)
        sigma2_i = fill_nan_with_mean(np.divide(p_hat*(1-p_hat),num_points,where=num_points>0))
        tau2_hat = np.nanvar(stat_history)
        B_i = sigma2_i/(tau2_hat+sigma2_i)

        stat_history[stat_history!=stat_history] = p_hat
        df[col+'_EM'] = df[col]+B_i*(p_hat-df[col])
        print(col, p_hat)
    return df # ok if p_hats don't add up because they're avg of averages


'''
use validate stats before calling statsClass.update() method
'''
def is_valid(arr):
    return not np.isnan(arr).any()

'''
collects 12-month s/r stats relative to historical opponents
columns '52_s_adj','52_r_adj' represent how well a player
performs above average
'''
def generate_52_adj_stats(df,start_ind=0):
    players_stats = {}
    match_52_stats = np.zeros([2,len(df),2]) # 2x1 arrays for x_i, x_j's 12-month s/r performance

    w_l = ['w','l']
    for i, row in df.loc[start_ind:].iterrows():
        surface = row['surface']
        date = row['match_year'],row['match_month']
        avg_52_s, avg_52_r = row['avg_52_s'],row['avg_52_r']
        match_stats = [[],[]]

        # add new players to the dictionary
        for k,label in enumerate(w_l):
            if row[label+'_name'] not in players_stats:
                players_stats[row[label+'_name']] = adj_stats_52(date)

        # store pre-match adj stats
        for k,label in enumerate(w_l):
            players_stats[row[label+'_name']].set_month(date)

            # fill in player's adjusted stats prior to start of match
            match_52_stats[k][i] = players_stats[row[label+'_name']].adj_sr
            # update serving stats if not null
            if validate(row, label):
                sv_stats = (row[label+'_swon'],row[label+'_svpt'],row[label+'_rwon'],row[label+'_rpt'])


                # TODO: this is the troublesome line... could be extracting nan value from opponent
                # TODO: also rewrite this so it's readable (plus with arrays not obvious at)
                opp_r_ablty = players_stats[row[w_l[1-k]+'_name']].adj_sr[1] + avg_52_r
                opp_s_ablty = players_stats[row[w_l[1-k]+'_name']].adj_sr[0] + avg_52_s
                opp_stats = (opp_r_ablty * row[label + '_svpt'], opp_s_ablty * row[label + '_rpt'])
                match_stats[k] = sv_stats + opp_stats

        # update players' adjusted scores based on pre-match adjusted ratings
        for k,label in enumerate(w_l):
            # if is_valid(match_stats):
            if validate(row, label) and is_valid(match_stats):
                players_stats[row[label+'_name']].update(date,match_stats[k])

    for k,label in enumerate(w_l):
        df[label+'_52_s_adj'] = match_52_stats[k][:,0]
        df[label+'_52_r_adj'] = match_52_stats[k][:,1]
    return df


'''
generate delta between two players relative to shared opponent
delta_i^AB = (spw(A, C_i) - (1 - rpw(A, C_i))) - (spw(B, C_i) - (1 - rpw(B, C_i)))
'''
def generate_delta(p1_stats, p2_stats):
    p1_s_pct, p1_r_pct = p1_stats[0]/float(p1_stats[1]), p1_stats[2]/float(p1_stats[3])
    p2_s_pct, p2_r_pct = p2_stats[0]/float(p2_stats[1]), p2_stats[2]/float(p2_stats[3])
    return (p1_s_pct - (1 - p1_r_pct)) - (p2_s_pct - (1 - p2_r_pct))

'''
return true if total service/return points both greater than zero
'''
def has_stats(last_year_stats):
    return last_year_stats[1] > 0 and last_year_stats[3] > 0

'''
get opponents who have played a match in the past 12 months (more than 0 points)
'''
def get_opponents(player_d, player_name):
    historical_opponents = player_d[player_name].history.keys()
    return [opp for opp in historical_opponents if has_stats(player_d[player_name].history[opp])]

'''
compute serve/return parameters, given their common opponent history
'''
def generate_commop_params(player_d, player1, player2):
    p1_opponents, p2_opponents = get_opponents(player_d, player1), get_opponents(player_d, player2)
    common_opponents = np.intersect1d(p1_opponents, p2_opponents)
    if len(common_opponents) == 0:
        return [0]

    match_deltas = np.zeros(len(common_opponents))
    for i, comm_op in enumerate(common_opponents):
        p1_match_stats = player_d[player1].history[comm_op]
        p2_match_stats = player_d[player2].history[comm_op]
        comm_op_delta = generate_delta(p1_match_stats, p2_match_stats)
        match_deltas[i] = comm_op_delta
        if np.isnan(comm_op_delta):
            print('nan here: ', p1_match_stats, p2_match_stats, comm_op)

    overall_delta = np.mean(match_deltas)
    if np.isnan(overall_delta):
        print('nan, match_deltas: ', match_deltas)
    return match_deltas

'''
collect historical s/r common-opponent performance by player
'''
def generate_commop_stats(df, start_ind):
    player_d = {}
    match_52_stats = np.zeros([2,len(df), 2])
    match_probs = np.zeros([len(df)])

    w_l = ['w','l']
    for i, row in df.loc[start_ind:].iterrows():
        for k, label in enumerate(w_l):
            opponent_name = row[w_l[1-k]+'_name']
            if row[label+'_name'] not in player_d:
                player_d[row[label+'_name']] = commop_stats()

            if validate(row, label):
                match_stats = (row[label+'_swon'],row[label+'_svpt'],row[w_l[1-k]+'_svpt']-\
                                row[w_l[1-k]+'_swon'],row[w_l[1-k]+'_svpt'])
                player_d[row[label+'_name']].update(match_stats, opponent_name)

        # can compute common-opponent stats after current match stats inputted
        if row['match_year'] >= COMMOP_START_YEAR: # start at COMMOP_START_YEAR, computationally intensive
            match_deltas = generate_commop_params(player_d, row['w_name'], row['l_name'])
            overall_delta = np.mean(match_deltas)
            w_s_pct, w_r_pct = (.6 + overall_delta/2), (.4 + overall_delta/2)

            match_52_stats[0][i] = [w_s_pct, w_r_pct]
            match_52_stats[1][i] = [1 - w_r_pct, 1 - w_s_pct]

            iterated_match_probs = [
                np.mean([
                    matchProb(.6 + match_delta, .4),
                    matchProb(.6, .4 + match_delta)
                ])
                for match_delta in match_deltas
            ]
            match_probs[i] = np.mean(iterated_match_probs)

    for k,label in enumerate(w_l):
        df[label+'_commop_s_pct'] = match_52_stats[k][:,0]
        df[label+'_commop_r_pct'] = match_52_stats[k][:,1]

    df['w_commop_match_prob'] = match_probs
    return df


'''
collect yearly tournament serve averages for 'f_av'
in Barnette-Clark equation
'''
def generate_tny_stats(df,start_ind=0):
    tny_stats = {}
    tny_52_stats = np.zeros(len(df))
    for i, row in df.loc[start_ind:].iterrows():
        if row['tny_name']=='Davis Cup':
            continue

        year,t_id = row['tny_id'].split('-')
        year = int(year)
        match_stats = (row['w_swon']+row['l_swon'],row['w_svpt']+row['l_svpt'])
        # handle nan cases, provide tny_stats if possible
        if row['w_swon']!=row['w_swon']:
            if t_id in tny_stats:
                if year-1 in tny_stats[t_id].historical_avgs:
                    swon,svpt = tny_stats[t_id].historical_avgs[year-1]
                    tny_52_stats[i] = swon/float(svpt)
            continue
        # create new object if needed, then update
        elif t_id not in tny_stats:
            tny_stats[t_id] = tny_52(year)
        tny_52_stats[i] = tny_stats[t_id].update(year,match_stats)

    df['tny_stats'] = tny_52_stats
    return df

'''
approximate inverse elo-->s_pct calculator
'''
def elo_induced_s(prob, s_total):
    s0 = s_total/2
    diff = s_total/4
    current_prob = .5

    while abs(current_prob-prob) > EPSILON:
        if current_prob < prob:
            s0 += diff
        else:
            s0 -= diff
        diff /= 2
        current_prob = matchProb(s0,1-(s_total-s0))
    return s0, s_total-s0


'''
import to set s_total with EM-normalized percentages
'''
def generate_bc_stats_elo_induced(df,col,start_ind=0):
    df['s_total'] = df['p0_s_kls_EM'] + df['p1_s_kls_EM']
    induced_s = np.zeros([len(df),2])
    for i, row in df.loc[start_ind:].iterrows():
        induced_s[i] = elo_induced_s(row[col+'_prob'],row['s_total'])
    df['p0_s_kls_' + col] = induced_s[:,0]
    df['p1_s_kls_' + col] = induced_s[:,1]

    del df['s_total']
    return df

def format_pbp_df(df,tour='atp'):
    df['w_name'] = np.where(df['winner'] == 0, df['server1'], df['server2'])
    df['l_name'] = np.where(df['winner'] == 0, df['server2'], df['server1'])
    df['w_name'] = [normalize_name(x,tour=tour) for x in df['w_name']]
    df['l_name'] = [normalize_name(x,tour=tour) for x in df['l_name']]
    df['date'] = pd.to_datetime(df['date'])
    df['match_year'] = [x.year for x in df['date']]
    df['match_month'] = [x.month for x in df['date']]
    df['date'] = [x.date() for x in df['date']]
    df['score'] = [re.sub(r"[\(\[].*?[\)\]]", "", s) for s in df['score']]
    return df

def connect_match_and_pbp_dfs(match_df,pbp_df,col_d,player_cols,start_year=2009):
    pbp_dict = {}; winner_dict = {}
    for i in xrange(len(pbp_df)):
        key = pbp_df['w_name'][i] +' ' +  pbp_df['l_name'][i] + ' ' \
            + str(pbp_df['match_year'][i]) + ' ' + pbp_df['score'][i]
        key = key+' '+str(pbp_df['match_month'][i]) if key in col_d else key
        if key in pbp_dict:
            continue
        pbp_dict[key] = pbp_df['pbp'][i]
        winner_dict[key] = pbp_df['winner'][i]

    # in case of a collision (about 10 cases), I only take the first match with that key
    c = 0
    pbps,winners = [],[]
    info = {}

    match_df = match_df[match_df['match_year']>=start_year]
    for i in match_df.index:
        key = match_df['w_name'][i] +' ' +  match_df['l_name'][i] + ' ' \
            +str(match_df['match_year'][i])+' '+match_df['score'][i]
        key = key+' '+str(match_df['match_month'][i]) if key in col_d else key
        if key in pbp_dict:
            c += 1
            pbps.append(pbp_dict[key])
            winners.append(winner_dict[key])
            if key in info:
                pbps[-1] = 'None'; winners[-1] = 'None'
                print('collision');
                print(key + ' ' + str(match_df['match_month'][i]))
            info[key] = 1
        else:
            pbps.append('None')
            # we'll just make 'winner' a random 0 or 1 for now
            winners.append(np.random.choice([0,1]))
    print(c)
    match_df['pbp'] = pbps
    match_df['winner'] = winners

    #df = match_df[match_df['pbp']!='NA']
    #cols = df.columns.drop(['loser_id','winner_id'])
    df = match_df[match_df.columns.drop(['loser_id','winner_id'])]
    df = df.reset_index(drop=True)

    # # change w,l TO p0,p1
    # for col in player_cols:
    #     df['p0'+col] = [df['l'+col][i] if df['winner'][i] else df['w'+col][i] for i in xrange(len(df))]
    #     df['p1'+col] = [df['w'+col][i] if df['winner'][i] else df['l'+col][i] for i in xrange(len(df))]

    # # add s/r pct columns
    # p_hat = np.sum([df['p0_52_swon'],df['p1_52_swon']])/np.sum([df['p0_52_svpt'],df['p1_52_svpt']])
    # for label in ['p0','p1']:
    #     df[label+'_s_pct'] = [p_hat if x==0 else x for x in np.nan_to_num(df[label+'_52_swon']/df[label+'_52_svpt'])]
    #     df[label+'_r_pct'] = [1-p_hat if x==0 else x for x in np.nan_to_num(df[label+'_52_rwon']/df[label+'_52_rpt'])]
    #     df[label+'_sf_s_pct'] = [p_hat if x==0 else x for x in np.nan_to_num(df[label+'_sf_52_swon']/df[label+'_sf_52_svpt'])]
    #     df[label+'_sf_r_pct'] = [1-p_hat if x==0 else x for x in np.nan_to_num(df[label+'_sf_52_rwon']/df[label+'_sf_52_rpt'])]

    # df['elo_diff'] = [df['p0_elo'][i] - df['p1_elo'][i] for i in xrange(len(df))]
    # df['sf_elo_diff'] = [df['p0_sf_elo'][i] - df['p1_sf_elo'][i] for i in xrange(len(df))]
    # df['tny_name'] = [s if s==s else 'Davis Cup' for s in df['tny_name']]
    return df
