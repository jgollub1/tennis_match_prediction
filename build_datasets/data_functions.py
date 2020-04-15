# see if it works with this commented out???

import sys
sys.path.insert(0,'/Users/jacobgollub/Desktop/college/research/pbp_explorations/scripts/sackmann')
import tennisGameProbability,tennisMatchProbability,tennisSetProbability,tennisTiebreakProbability
from tennisMatchProbability import matchProb

import numpy as np
import pandas as pd
import elo_538 as elo
from helper_functions import adj_stats_52,stats_52,tny_52,normalize_name
from sklearn import linear_model
import re
import datetime

def format_match_df(df,ret_strings=[],abd_strings=[]):
    df['pbp'] = [None]*len(df)
    df = df.rename(columns={'winner_name':'w_name','loser_name':'l_name'})
    grand_slam_d = dict(zip(['Australian Open','Roland Garros','Wimbledon','US Open'],[1]*4))
    df['is_gs'] = [name in grand_slam_d for name in df['tny_name']]

    df['newwdelta1'] = df['w_delta1']
    df['newwdelta2'] = df['w_delta2']
    df = df[~df['newwdelta1'].isin(['#VALUE!'])]
    df = df[~df['newwdelta2'].isin(['#VALUE!'])]

    nametest = df['newwdelta1'].astype(float)
    nametest1 = df['newwdelta2']

    # df['minusnewwdelta1'] = 1 - nametest
    # df['minusnewwdelta2'] = 1 - nametest1

    df['minusnewwdelta1'] = nametest
    df['minusnewwdelta2'] = nametest1

    # Get dates into the same format
    print(len(df))
    df = df[df['tny_date'].notna()]
    print(len(df))
    df['tny_date'] = [datetime.datetime.strptime(str(x), "%Y.%m.%d").date() for x in df['tny_date']]
    df['match_year'] = [x.year for x in df['tny_date']]
    df['match_month'] = [x.month for x in df['tny_date']]
    # df['score'] = [re.sub(r"[\(\[].*?[\)\]]", "", str(s)) for s in df['score']] # str(s) fixes any nans
    # df['score'] = ['RET' if 'RET' in s else s for s in df['score']]
    # df['w_swon'] = [df['w_1stWon'][i]+df['w_2ndWon'][i] for i in xrange(len(df))]
    # df['l_swon'] = [df['l_1stWon'][i]+df['l_2ndWon'][i] for i in xrange(len(df))]
    # df['w_rwon'] = df['l_svpt'] - df['l_swon']
    # df['l_rwon'] = df['w_svpt'] - df['w_swon']
    # df['w_rpt'] = df['l_svpt']
    # df['l_rpt'] = df['w_svpt']

    # get rid of leading 0s in tny_id
    df['tny_id'] = ['-'.join(t.split('-')[:-1] + [t.split('-')[-1][1:]]) if t.split('-')[-1][0]=='0' else t \
                                    for t in df['tny_id']]

    # get rid of matches involving a retirement
    # df['score'] = ['ABN' if score.split(' ')[-1] in ('abandoned','ABN','ABD','DEF','def','unfinished','Walkover') \
    #                             else score for score in df['score']]
    ret_strings = ['ABN','DEF','In Progress','RET','W/O',' RET',' W/O','nan','walkover']
    ret_d = dict(zip(ret_strings,[1]*len(ret_strings)))
    # df = df.loc[[i for i in range(len(df)) if df['score'][i] not in ret_d]]
    df = df.sort_values(by=['tny_date','tny_name','match_num'], ascending=True).reset_index()
    del df['index']; return df

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

def generate_elo_stephanie(df,counts):
    players_list = np.union1d(df.w_name, df.l_name)
    players_elo = dict(zip(list(players_list), [elo.Rating() for __ in range(len(players_list))]))
    surface_elo = {}
    for surface in ('Hard','Clay','Grass'):
        surface_elo[surface] = dict(zip(list(players_list), [elo.Rating() for __ in range(len(players_list))])) 

    elo_1s, elo_2s = [],[]
    surface_elo_1s, surface_elo_2s = [],[]
    elo_obj = elo.Elo_Rater()

    k1, k2 = 5.3, 16

    # k1, k2 = 2.5, 24

    # update player elo from every recorded match
    for i, row in df.iterrows():
        surface = row['surface']; is_gs = row['is_gs']; tny_name = row['tny_name']; tny_round_name = row['tourney_round_name']
        #delta1 = row['newwdelta1'] ; delta2 = row['newwdelta2']
        delta1 = row['minusnewwdelta1'] ; delta2 = row['minusnewwdelta2']
        # append elos, rate, update
        w_elo,l_elo = players_elo[row['w_name']],players_elo[row['l_name']]
        
        elo_1s.append(w_elo.value);elo_2s.append(l_elo.value)
        
        elo_obj.rate_1vs1_stephanie(w_elo,l_elo, k1, k2, delta1, delta2, is_gs, counts, tny_name, tny_round_name )

        surface_elo_1s.append(surface_elo[surface][row['w_name']].value if surface in ('Hard','Clay','Grass') else w_elo.value)
        surface_elo_2s.append(surface_elo[surface][row['l_name']].value if surface in ('Hard','Clay','Grass') else l_elo.value)
        if surface in ('Hard','Clay','Grass'):
            new_elo1, new_elo2 = elo_obj.rate_1vs1_stephanie(surface_elo[surface][row['w_name']],surface_elo[surface][row['l_name']], k1, k2, delta1, delta2, is_gs, counts, tny_name, tny_round_name)

    # add columns
    if counts:
        df['w_elo_538'], df['l_elo_538'] = elo_1s, elo_2s
        df['w_sf_elo_538'], df['l_sf_elo_538'] = surface_elo_1s, surface_elo_2s
    else:
        df['w_elo'], df['l_elo'] = elo_1s, elo_2s
        df['w_sf_elo'], df['l_sf_elo'] = surface_elo_1s, surface_elo_2s
    return df


# takes in a dataframe of matches in atp/wta format and returns the dataframe with elo columns
def generate_elo(df,counts_i):
    players_list = np.union1d(df.w_name, df.l_name)
    players_elo = dict(zip(list(players_list), [elo.Rating() for __ in range(len(players_list))]))
    surface_elo = {}
    for surface in ('Hard','Clay','Grass'):
        surface_elo[surface] = dict(zip(list(players_list), [elo.Rating() for __ in range(len(players_list))])) 

    elo_1s, elo_2s = [],[]
    surface_elo_1s, surface_elo_2s = [],[]
    elo_obj = elo.Elo_Rater()

    # update player elo from every recorded match
    for i, row in df.iterrows():
        surface = row['surface']; is_gs = row['is_gs']; tny_name = row['tny_name']
        # append elos, rate, update
        w_elo,l_elo = players_elo[row['w_name']],players_elo[row['l_name']]
        elo_1s.append(w_elo.value);elo_2s.append(l_elo.value)

        # original
        # elo_obj.rate_1vs1(w_elo,l_elo,is_gs,counts=counts_i)

        # adjust elo update based on other scalers
        # elo_obj.rate_1vs1(w_elo,l_elo,is_gs, counts=counts_i, tny_name=row['tny_name'], tny_round_name=row['tourney_round_name'],\
        # w_pts_won=row['w_pt_won'], l_pts_won=row['l_pt_won'], pts_total=row['pt_total'],\
        # w_s1_pts_won=row['winner_service_points_won'], l_s1_pts_won=row['loser_service_points_won'],\
        # w_s1_pts_total=row['winner_service_points_total'], l_s1_pts_total=row['loser_service_points_total'])
        
        elo_obj.rate_1vs1(w_elo,l_elo,is_gs, counts=counts_i, tny_name=row['tny_name'], tny_round_name=row['tourney_round_name'])

        surface_elo_1s.append(surface_elo[surface][row['w_name']].value if surface in ('Hard','Clay','Grass') else w_elo.value)
        surface_elo_2s.append(surface_elo[surface][row['l_name']].value if surface in ('Hard','Clay','Grass') else l_elo.value)
        if surface in ('Hard','Clay','Grass'):
            new_elo1, new_elo2 = elo_obj.rate_1vs1(surface_elo[surface][row['w_name']],surface_elo[surface][row['l_name']],is_gs,counts=counts_i)

    # add columns
    if counts_i:
        df['w_elo_538'], df['l_elo_538'] = elo_1s, elo_2s
        df['w_sf_elo_538'], df['l_sf_elo_538'] = surface_elo_1s, surface_elo_2s
    else:
        df['w_elo'], df['l_elo'] = elo_1s, elo_2s
        df['w_sf_elo'], df['l_sf_elo'] = surface_elo_1s, surface_elo_2s
    
    return df


def generate_52_stats(df,start_ind):
    players_stats = {}
    start_date = (df['match_year'][start_ind],df['match_month'][start_ind])
    avg_stats = stats_52(start_date)
    # set as prior so first row is not nan
    avg_stats.update(start_date,(6.4,10,3.6,10))
    # array w/ 2x1 arrays for each player's 12-month serve/return performance
    match_52_stats = np.zeros([2,len(df),4])
    avg_52_stats = np.zeros([len(df),4])

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
            match_52_stats[k][i] = np.sum(players_stats[row[label+'_name']].last_year,axis=0)
            # update serving stats if not null
            if row[label+'_swon']==row[label+'_swon'] and row[label+'_svpt']==row[label+'_svpt']:    
                match_stats = (row[label+'_swon'],row[label+'_svpt'],row[w_l[1-k]+'_svpt']-\
                                row[w_l[1-k]+'_swon'],row[w_l[1-k]+'_svpt'])
                players_stats[row[label+'_name']].update(date,match_stats)
                avg_stats.update(date,match_stats)

        if surface not in ('Hard','Clay','Grass'):
            continue
        # repeat above process for surface-specific stats
        s_avg_stats[surface].set_month(date)
        s_avg_52_stats[i] = np.sum(s_avg_stats[surface].last_year,axis=0)
        for k,label in enumerate(w_l):
            if row[label+'_name'] not in s_players_stats[surface]:
                s_players_stats[surface][row[label+'_name']] = stats_52(date)
            
            # store serving stats prior to match, from current month
            s_players_stats[surface][row[label+'_name']].set_month(date)
            s_match_52_stats[k][i] = np.sum(s_players_stats[surface][row[label+'_name']].last_year,axis=0)
            # update serving stats if not null
            if row[label+'_swon']==row[label+'_swon'] and row[label+'_svpt']==row[label+'_svpt']:    
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
        df['avg_52_s'] = np.divide(avg_52_stats[:,0],avg_52_stats[:,1])
        df['avg_52_r'] = np.divide(avg_52_stats[:,2],avg_52_stats[:,3])
        df['sf_avg_52_s'] = np.divide(s_avg_52_stats[:,0],s_avg_52_stats[:,1])
        df['sf_avg_52_r'] = np.divide(s_avg_52_stats[:,2],s_avg_52_stats[:,3])
    return df

def generate_52_adj_stats(df,start_ind=0):
    players_stats = {}
    # array w/ 2x1 arrays for each player's 12-month serve/return performance
    match_52_stats = np.zeros([2,len(df),2])
    
    w_l = ['w','l']
    for i, row in df.loc[start_ind:].iterrows():
        surface = row['surface']  
        date = row['match_year'],row['match_month']
        avg_52_s,avg_52_r = row['avg_52_s'],row['avg_52_r']
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
            if row[label+'_swon']==row[label+'_swon'] and row[label+'_svpt']==row[label+'_svpt']:
                sv_stats = (row[label+'_swon'],row[label+'_svpt'],row[label+'_rwon'],row[label+'_rpt'])
                opp_r_ablty = players_stats[row[w_l[1-k]+'_name']].adj_sr[1]+avg_52_r
                opp_s_ablty = players_stats[row[w_l[1-k]+'_name']].adj_sr[0]+avg_52_s
                opp_stats = (opp_r_ablty*row[label+'_svpt'], opp_s_ablty*row[label+'_rpt'])
                match_stats[k] = sv_stats+opp_stats

                
        # update players' adjusted scores based on pre-match adjusted ratings
        for k,label in enumerate(w_l):
            if row[label+'_swon']==row[label+'_swon'] and row[label+'_svpt']==row[label+'_svpt']:
                players_stats[row[label+'_name']].update(date,match_stats[k])
            
    for k,label in enumerate(w_l):
        df[label+'_52_s_adj'] = match_52_stats[k][:,0]
        df[label+'_52_r_adj'] = match_52_stats[k][:,1]

    return df

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

# have to correct the winner problem
def connect_df(match_df,pbp_df,col_d,player_cols,start_year=2009):
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
                print 'collision'; print key + ' ' + str(match_df['match_month'][i])
            info[key] = 1
        else:
            pbps.append('None')
            # we'll just make 'winner' a random 0 or 1 for now
            winners.append(np.random.choice([0,1]))
    print c
    match_df['pbp'] = pbps
    match_df['winner'] = winners

    #df = match_df[match_df['pbp']!='NA']
    #cols = df.columns.drop(['loser_id','winner_id'])
    df = match_df[match_df.columns.drop(['loser_id','winner_id'])]
    df = df.reset_index(drop=True)

    # change w,l TO p0,p1
    for col in player_cols:
        df['p0'+col] = [df['l'+col][i] if df['winner'][i] else df['w'+col][i] for i in xrange(len(df))]
        df['p1'+col] = [df['w'+col][i] if df['winner'][i] else df['l'+col][i] for i in xrange(len(df))]

    # add s/r pct columns
    p_hat = np.sum([df['p0_52_swon'],df['p1_52_swon']])/np.sum([df['p0_52_svpt'],df['p1_52_svpt']])
    for label in ['p0','p1']:
        df[label+'_s_pct'] = [p_hat if x==0 else x for x in np.nan_to_num(df[label+'_52_swon']/df[label+'_52_svpt'])]
        df[label+'_r_pct'] = [1-p_hat if x==0 else x for x in np.nan_to_num(df[label+'_52_rwon']/df[label+'_52_rpt'])]
        df[label+'_sf_s_pct'] = [p_hat if x==0 else x for x in np.nan_to_num(df[label+'_sf_52_swon']/df[label+'_sf_52_svpt'])]
        df[label+'_sf_r_pct'] = [1-p_hat if x==0 else x for x in np.nan_to_num(df[label+'_sf_52_rwon']/df[label+'_sf_52_rpt'])]

    df['elo_diff'] = [df['p0_elo'][i] - df['p1_elo'][i] for i in xrange(len(df))]
    df['sf_elo_diff'] = [df['p0_sf_elo'][i] - df['p1_sf_elo'][i] for i in xrange(len(df))]
    df['tny_name'] = [s if s==s else 'Davis Cup' for s in df['tny_name']]
    return df

def generate_logit_probs(df,cols,col_name):
    lm = linear_model.LogisticRegression(fit_intercept = True)
    df_train = df[df['match_year'].isin([2011,2012,2013])]
    df_train = df_train[df_train['winner'].isin([0,1])]
    df_train['winner'] = df_train['winner'].astype(int)
    lm.fit(df_train[cols].values.reshape([df_train.shape[0],len(cols)]),np.asarray(df_train['winner']))
    print 'cols: ', cols
    print 'lm coefficients: ', lm.coef_
    df[col_name] = lm.predict_proba(df[cols].values.reshape([df.shape[0],len(cols)]))[:,0]
    return df

# approximation inverse elo-->s_pct calculator
def elo_induced_s(prob,s_total):
    s0 = s_total/2
    current_prob = .5
    diff = s_total/4
    while abs(current_prob-prob)>.001:
        if current_prob < prob:
            s0 += diff
        else:
            s0 -= diff
        diff /= 2
        current_prob = matchProb(s0,1-(s_total-s0))
    return s0,s_total-s0

# import to set s_total with JS-normalized percentages
def generate_elo_induced_s(df,col,start_ind=0):
    df['s_total'] = df['p0_s_kls_JS'] + df['p1_s_kls_JS']
    induced_s = np.zeros([len(df),2])
    for i, row in df.loc[start_ind:].iterrows():
        induced_s[i] = elo_induced_s(row[col+'_prob'],row['s_total'])
    df['p0_s_kls_'+col] = induced_s[:,0]
    df['p1_s_kls_'+col] = induced_s[:,1]
    return df

def generate_JS_stats(df,cols):
    #### James-Stein estimators for 52-week serve and return percentages ####
    # calculate B_i coefficients for each player in terms of service points
    for col in cols:
        stat_history = np.concatenate([df['p0_'+col],df['p1_'+col]],axis=0)
        n = len(stat_history)/2
        group_var = np.var(stat_history)
        num_points = np.concatenate([df['p0_52_svpt'],df['p1_52_svpt']]) if '_s_' in col \
                    else np.concatenate([df['p0_52_rpt'],df['p1_52_rpt']])
        p_hat = np.mean(stat_history)
        sigma2_i = np.divide(p_hat*(1-p_hat),num_points,where=num_points>0)
        tau2_hat = np.nanvar(stat_history)
        #print 'col: ', col
        #print 'sigma2, tau2:', sigma2_i, tau2_hat
        # print col
        # print p_hat, tau2_hat,np.nanvar(num_points)
        B_i = sigma2_i/(tau2_hat+sigma2_i)
        df['B_'+col+'_i0_sv'],df['B_'+col+'_i1_sv'] = B_i[:n],B_i[n:]

        stat_history[stat_history!=stat_history] = p_hat
        group_var = np.var(stat_history)
        df['p0_'+col+'_JS'] = df['p0_'+col]+df['B_'+col+'_i0_sv']*(p_hat-df['p0_'+col])
        df['p1_'+col+'_JS'] = df['p1_'+col]+df['B_'+col+'_i1_sv']*(p_hat-df['p1_'+col])
        print col, p_hat


    # repeat for surface stats and overall stats
    p_hat = np.sum([df['p0_52_swon'],df['p1_52_swon']])/np.sum([df['p0_52_svpt'],df['p1_52_svpt']])
    for sv in ['','sf_']:
        s_history = np.concatenate([df['p0_'+sv+'52_swon']/df['p0_'+sv+'52_svpt'],\
                    df['p1_'+sv+'52_swon']/df['p1_'+sv+'52_svpt']],axis=0)
        n = len(s_history)/2
        group_var = np.var(s_history)
        s_points = np.concatenate([df['p0_'+sv+'52_svpt'],df['p1_'+sv+'52_svpt']])
        sigma2_i = np.divide(p_hat*(1-p_hat),s_points,where=s_points>0)
        tau2_hat = np.nanvar(s_history)
        B_i = sigma2_i/(tau2_hat+sigma2_i)
        df['B_'+sv+'i0_sv'],df['B_'+sv+'i1_sv'] = B_i[:n],B_i[n:]

        s_history[s_history!=s_history] = p_hat
        group_var = np.var(s_history)
        df['p0_'+sv+'s_pct_JS'] = df['p0_'+sv+'s_pct']+df['B_'+sv+'i0_sv']*(p_hat-df['p0_'+sv+'s_pct'])
        df['p1_'+sv+'s_pct_JS'] = df['p1_'+sv+'s_pct']+df['B_'+sv+'i1_sv']*(p_hat-df['p1_'+sv+'s_pct'])

        # repeat for return averages (slightly different tau^2 value)
        r_history = np.concatenate([df['p0_'+sv+'52_rwon']/df['p0_'+sv+'52_rpt'],\
                    df['p1_'+sv+'52_rwon']/df['p1_'+sv+'52_rpt']],axis=0)
        r_points = np.concatenate([df['p0_'+sv+'52_rpt'],df['p1_'+sv+'52_rpt']])
        sigma2_i = np.divide((1-p_hat)*p_hat,r_points,where=r_points>0)
        tau2_hat = np.nanvar(r_history)
        B_i = sigma2_i/(tau2_hat+sigma2_i)
        df['B_'+sv+'i0_r'],df['B_'+sv+'i1_r'] = B_i[:n],B_i[n:]

        r_history[r_history!=r_history] = 1-p_hat
        df['p0_'+sv+'r_pct_JS'] = r_history[:n]+df['B_'+sv+'i0_r']*((1-p_hat)-r_history[:n])
        df['p1_'+sv+'r_pct_JS'] = r_history[n:]+df['B_'+sv+'i1_r']*((1-p_hat)-r_history[n:])
    return df



