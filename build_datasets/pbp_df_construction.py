from helper_functions import *
from data_functions import *
import pandas as pd
import numpy as np
import time

# indicator to generate point-by-point for best-of-three matches (else: best-of-five)
BEST_OF_THREE = 0
FILE_NAME = 'elo_pbp_with_surface_10_23.csv'

if __name__=='__main__':
	df = pd.read_csv('../../my_data/'+FILE_NAME)
	del df['Unnamed: 0']

	# append . to end of pbp string to signify end of match
	df['pbp'] = [s + '.' if s[-1]!= '.' else s for s in df['pbp']]

	#tid_dict = dict(zip(df['match_id'],df['tny_id']))
	df['tny_name'] = [df['tny_name'][i]+' (3)' if df['tny_name'][i]=='Davis Cup' and \
	                  df['best_of'][i]==3 else df['tny_name'][i] for i in xrange(len(df))]

	gs_tourneys = ['Australian Open', 'Davis Cup', 'Roland Garros', 'US Open', 'Wimbledon']
	df_pbp3 = df[~df['tny_name'].isin(gs_tourneys)]
	df_pbp5 = df[df['tny_name'].isin(gs_tourneys)]
	df_pbp3=df_pbp3.reset_index(drop=True);df_pbp5=df_pbp5.reset_index(drop=True)

	start = time.clock()
	# specify which columns to carry over to point-by-point df
	cols = ['match_id','surface','elo_diff','sf_elo_diff','winner','p0_s_pct','p0_s_pct_JS',
	        'p1_s_pct','p1_s_pct_JS','p0_r_pct','p0_r_pct_JS','p1_r_pct','p1_r_pct_JS', 
	        'p0_s_kls','p1_s_kls','p0_s_kls_JS','p1_s_kls_JS','p0_sf_s_kls','p1_sf_s_kls', 
	        'p0_sf_s_kls_JS','p1_sf_s_kls_JS',
	        'p0_s_kls_adj','p1_s_kls_adj','p0_s_kls_adj_JS','p1_s_kls_adj_JS',
	        'p0_s_kls_elo', 'p1_s_kls_elo','p0_s_kls_logit_elo_538','p1_s_kls_logit_elo_538',
	        #'',
	        'tny_stats','best_of']
	enum_df = df_pbp3 if BEST_OF_THREE else df_pbp5
	df_pred = generate_df_2(enum_df,cols,0)
	df_pred = df_pred.reset_index(drop=True)
	print df_pred.columns
	print 'df ('+str(len(df_pred))+' points) generated in: ', time.clock()-start,'seconds'
	print 'computing break point statistics...'

	# break-point indicators
	breaks = [0]*len(df_pred)
	for i in xrange(len(df_pred)):
	    breaks[i] = break_point(df_pred['score'][i])
	df_pred['up_break_point'] = [a[0] for a in breaks]
	df_pred['down_break_point'] = [a[1] for a in breaks]
	df_pred['break_adv'] = [service_breaks(df_pred['score'][i]) for i in xrange(len(df_pred))]
	#df_pred['tny_id'] = [tid_dict[m_id] for m_id in df_pred['match_id']]
	df_pred['server'] = 1 - df_pred['server']
	# add a heuristic lead function
	df_pred['lead_margin'] = df_pred['sets_0']-df_pred['sets_1'] + (df_pred['games_0']-\
							df_pred['games_1'])/6. + (df_pred['points_0']-df_pred['points_1'])/24.
	best_of = '3' if BEST_OF_THREE else '5'
	df_pred.to_csv('../../my_data/feature_df_pbp'+best_of+'_10_23.csv')
	print 'feature_df_pbp'+best_of+'_10_23.csv saved to my_data'



