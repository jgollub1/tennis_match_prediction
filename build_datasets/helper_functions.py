# TO DO: 1) clean up, thoroughly, the existing code here
# 2) write enumerate_pbp, which will be handy if we use
# any future models that do rely on order
# get set_order is acting screwy

import math
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss,accuracy_score

# accept dates in (year,month); last_year contains last 12 month stats, most recent to least
class stats_52():
    def __init__(self,date):
        self.most_recent = date
        self.last_year = np.zeros([12,4])
        
    def time_diff(self,new_date,old_date):
        return 12*(new_date[0]-old_date[0])+(new_date[1]-old_date[1])

    def set_month(self,match_date):
        diff = self.time_diff(match_date,self.most_recent)
        if diff>=12:
            self.last_year = np.zeros([12,4])
        elif diff>0:
            self.last_year[diff:] = self.last_year[:12-diff]; self.last_year[:diff] = 0
        self.most_recent = match_date

    
    def update(self,match_date,match_stats):
        self.set_month(match_date)
        self.last_year[0] = self.last_year[0]+match_stats
        

# a similar class to store a tournament's serving averages from the previous year
class tny_52():
    def __init__(self,date):
        self.most_recent = date
        self.tny_stats = np.zeros([2,2])
        self.historical_avgs = {}
    
    def update(self,match_year,match_stats):
        diff = match_year-self.most_recent
        if diff>=2:
            self.tny_stats = np.zeros([2,2])
        elif diff==1:
            self.tny_stats[1] = self.tny_stats[0]; self.tny_stats[0]=0
        self.tny_stats[0] = self.tny_stats[0]+match_stats
        self.most_recent = match_year
        self.historical_avgs[match_year] = (self.tny_stats[0][0],self.tny_stats[0][1])
        return 0 if self.tny_stats[1][1]==0 else self.tny_stats[1][0]/float(self.tny_stats[1][1])


# v3.0 with smarter object construction
# use np.array to create arrays from lists; use np.concatenate to combine arrays
# figuring out the last three lines here made my function about four times faster...
def enumerate_pbp_2(s,columns,final_set_extend=0):
    # find the number of S,R,D,A characters and use this to initialize
    # all columns as npy arrays of this length
    length = len(s.replace('.','').replace('/','').replace(';',''))

    sub_matches = ['']; sub_sets = [0]
    t_points,p0_tp, p1_tp = [0,0],[0],[0]
    s_points,p0_sp, p1_sp = [0,0],[0],[0]
    sw_points,p0_swp, p1_swp = [0,0],[0],[0]
    points,p0_p, p1_p = [0,0],[0],[0]
    games,p0_g, p1_g = [0,0],[0],[0]
    sets,p0_s, p1_s = [0,0],[0],[0]
    server = 0; servers=[]
    
    # divides into s_new, array of games
    s = s.split(';'); s_new = []
    for i in range(len(s)):
        if '.' in s[i]:
            game_str = s[i].split('.')
            game_str[0] += '.'
            s_new += game_str
        else:
            s_new += [s[i]]

    # iterate through each game, a point at a time
    for i in range(len(s_new)-1):
        # update server; up_til_now is everything that has elapsed
        server = (server+1)%2 if i>0 else server
        up_til_now = (';'.join(s_new[:i])+';' if i>0 else ';'.join(s_new[:i])).replace('.;','.')

        # update points and sets if there is a tiebreaker
        if games[0]==games[1]==6 and not (p0_s==p1_s==2 and final_set_extend):
            t_server = server
            mini_games = s_new[i].split('/')
            for j in range(len(mini_games)):
                for l in range(len(mini_games[j])):
                    if mini_games[j][l]!='.':
                        winner = int(mini_games[j][l] in ('S','A') and t_server==1 or mini_games[j][l] in ('R','D') and t_server==0) 
                        points[winner]+=1
                        t_points[winner]+=1
                        if winner==server: sw_points[winner]+=1
                        s_points[server]+=1
                        sub_m = up_til_now+'/'.join(mini_games[:j])+'/'+mini_games[j][:l+1] if j>0 else up_til_now+mini_games[j][:l+1]
                        sub_matches.append(sub_m)
                        p0_p.append(points[0]);p1_p.append(points[1])
                        p0_tp.append(t_points[0]);p1_tp.append(t_points[1])
                        p0_g.append(games[0]);p1_g.append(games[1])
                        p0_s.append(sets[0]);p1_s.append(sets[1])
                        p0_swp.append(sw_points[0]);p1_swp.append(sw_points[1])
                        p0_sp.append(s_points[0]);p1_sp.append(s_points[1])
                        servers.append(t_server)
                    else:
                        sets[winner]+=1
                        points, games = [0,0], [0,0]
                        p0_s[-1],p1_s[-1]=sets[0],sets[1]
                        p0_g[-1],p1_g[-1]=games[0],games[1]
                        p0_p[-1],p1_p[-1]=points[0],points[1]
                        sub_matches[-1] += '.'
                t_server = 1 - t_server

        # otherwise 
        else:
            for k in range(len(s_new[i])):
                if s_new[i][k]=='/':
                    print 'ERROR; this should be TIEBREAK'
                    return 0

                if s_new[i][k]!='.':    
                    winner = int(s_new[i][k] in ('S','A') and server==1 or s_new[i][k] in ('R','D') and server==0) 
                    points[winner]+=1
                    t_points[winner]+=1
                    if winner==server: sw_points[winner]+=1
                    s_points[server]+=1
                    if k==len(s_new[i])-1:
                        sub_matches.append(up_til_now+s_new[i][:k+1]+';')
                        games[winner]+=1
                        points = [0,0]
                    else:
                        sub_matches.append(up_til_now+s_new[i][:k+1])

                    p0_p.append(points[0]);p1_p.append(points[1])
                    p0_tp.append(t_points[0]);p1_tp.append(t_points[1])
                    p0_g.append(games[0]);p1_g.append(games[1])
                    p0_s.append(sets[0]);p1_s.append(sets[1])
                    p0_swp.append(sw_points[0]);p1_swp.append(sw_points[1])
                    p0_sp.append(s_points[0]);p1_sp.append(s_points[1])
                    servers.append(server)

                # backtrack and update previous entries if we finished a set; reset points and games
                elif s_new[i][k]=='.':
                    sets[winner]+=1
                    points, games = [0,0], [0,0]
                    p0_s[-1],p1_s[-1]=sets[0],sets[1]
                    p0_g[-1],p1_g[-1]=games[0],games[1]
                    p0_p[-1],p1_p[-1]=points[0],points[1]
                    sub_matches[-1] += '.'
        
        points = [0,0]
    
    columns = np.repeat([columns],[len(servers)],axis=0)
    generated_cols = np.array([p0_s[:-1],p1_s[:-1],p0_g[:-1],p1_g[:-1],p0_p[:-1],p1_p[:-1],p0_tp[:-1],p1_tp[:-1],p0_swp[:-1],p0_sp[:-1],p1_swp[:-1],p1_sp[:-1],servers]).T
    return sub_matches[:-1], np.concatenate([columns,generated_cols],axis=1)

# leave this here so you can modify column names
# NOTE: it is best to keep all the arrays in a list and then concatenate outside the loop
# columns param. specifies which columns to feed into the new dataframe
def generate_df_2(df_pbp,columns,final_set_no_tb):
    pbps,dfs = [0]*len(df_pbp),[0]*len(df_pbp)
    for i in xrange(len(df_pbp)):
        info = [df_pbp[col][i] for col in columns]
        a,b = enumerate_pbp_2(df_pbp['pbp'][i],info,final_set_no_tb)
        pbps[i],dfs[i] = a, np.asarray(b)

    df = pd.DataFrame(np.concatenate(dfs))
    df.columns = columns + ['sets_0','sets_1','games_0',\
                  'games_1','points_0','points_1','tp_0','tp_1','p0_swp','p0_sp','p1_swp','p1_sp','server']
    df['score'] = np.concatenate(pbps)
    df['in_lead'] = in_lead(df) 
    return df


# optimized function to check who leads, combining boolean indices and functions
def in_lead(df):
    c = np.array(df[['sets_0','games_0','points_0']]) - np.array(df[['sets_1','games_1','points_1']])
    set_d,game_d,point_d = c.T
    leads = np.zeros(len(c))
    set_ind = np.where(set_d!=0)[0]
    game_ind = np.logical_and(set_d==0, game_d!=0).nonzero()[0]
    point_ind = np.logical_and(set_d==0, game_d==0).nonzero()[0]
    leads[set_ind] = set_d[set_ind]>0
    leads[game_ind] = game_d[game_ind]>0
    leads[point_ind] = point_d[point_ind]>0
    return leads
# def generate_df_2(df_pbp,final_set_no_tb):
#     print 'hi'
#     dfs = [0]*len(df_pbp)
#     for i in xrange(len(df_pbp)):
#         info = [df_pbp['match_id'][i],df_pbp['elo_diff'][i],df_pbp['s_elo_diff'],df_pbp['winner'][i]]
#         dfs[i] = np.asarray(enumerate_pbp_2(df_pbp['pbp'][i],info,final_set_no_tb)).T
#     df = pd.DataFrame(np.concatenate(dfs))
#     df.columns = ['match_id','elo_diff','s_elo_diff','winner','score','server','sets_0','sets_1','games_0',\
#                   'games_1','points_0','points_1','tp_0','tp_1','p0_swp','p0_sp','p1_swp','p1_sp']
#     return df


# functions used to parse point-by-point tennis data
def simplify(s):
    s=s.replace('A','S');s=s.replace('D','R')
    sets = s.split('.')
    literal_s=''
    for k in range(len(sets)):
        # set server to 0 or 1 at beginning of set, keeping track of all transitions
        server = 0 if k==0 else next_server    
        games = sets[k].split(';');length = len(games)
        # update length of games (service switches) if there is tiebreak
        if length > 12:
            games = games[:-1] + games[-1].split('/')
            next_server = (server+1)%2
        else:
            next_server = (server + len(games))%2
        # now, iterate through every switch of serve
        for game in games:
            game = game.replace("S",str(server))
            game = game.replace("R",str((server+1)%2))
            literal_s += game
            server =(server+1)%2
    return literal_s

def service_breaks(s):
    ## return the service break advantage ##
    s=s.replace('A','S');s=s.replace('D','R')
    all_sets = s.split('.'); p1_games, p2_games = 0,0
    for k in range(len(all_sets)):
        # set server to 0 or 1 at beginning of set, keeping track of all transitions
        server = 0 if k==0 else next_server 
        games = all_sets[k].split(';');length = len(games)
        next_server = (server+1)%2 if length>12 else (server + len(games))%2   
        if k==len(all_sets)-1:
            completed_games = all_sets[k].split(';')[:-1]
            for i in range(len(completed_games)):
                if i!=0: 
                    server = (server+1)%2 
                game = completed_games[i]
                if server==0 and game[-1]=='S' or server==1 and game[-1]=='R':
                    p1_games += 1
                else:
                    p2_games += 1
        next_server = (server+1)%2 if length > 12 else (server + length)%2
    server = (server+1)%2
    if server==0:
        break_adv = math.ceil(float(p1_games-p2_games)/2)
    else:
        break_adv = math.ceil(float(p1_games-p2_games-1)/2)
    return int(break_adv)

def find_pattern(s,pattern):
    # invert the pattern so as to count occurrences for second player
    inv_pattern = pattern.replace('0','x')
    inv_pattern = inv_pattern.replace('1','0')
    inv_pattern = inv_pattern.replace('x','1')
    pattern = list(pattern);inv_pattern = list(inv_pattern)
    
    times = [0,0]
    # convert string to list of chars so we can reassign characters after using them
    literal_s = list(s)
    # now, just find triples of '000' and '111' (for players 1 and 2 respectively)
    for i in range(len(literal_s) - len(pattern) + 1):
        if literal_s[i:i + len(pattern)]==pattern:
            times[0]+=1
            literal_s[i+len(pattern)-1] = 'x'
        elif literal_s[i:i + len(pattern)]==inv_pattern:
            times[1]+=1
            literal_s[i+len(pattern)-1] = 'x'  
    return times


def get_set_order(s):
    s=s.replace('A','S');s=s.replace('D','R')
    # split the string on '.' and count sets up to the second to last entry
    # (if the substring ends on a '.' the last element will be '')
    completed_sets = s.split('.')[:-1]
    sets = ''
    for k in range(len(completed_sets)):
        # set server to 0 or 1 at beginning of set, keeping track of all transitions
        server = 0 if k==0 else next_server              
        games = completed_sets[k].split(';');length = len(games)
        # update length of games (service switches) if there is tiebreak
        if length > 12:
            games = games[:-1] + games[-1].split('/')
            next_server = (server+1)%2
        else:
            next_server = (server + len(games))%2
        final_server = (server + len(games) - 1)%2
        # award set to the player who won the last point of the set
        if final_server==0 and games[-1][-1]=='S':
            sets += '0'
        elif final_server==1 and games[-1][-1]=='R':
            sets += '0'
        else:
            sets += '1'
    return sets

# gets game order of entire match, with sets separated by periods
def get_game_order(s):
    s=s.replace('A','S');s=s.replace('D','R')  
    # last entry in this will be '' if we split at the end of a set
    all_sets = s.split('.')[:-1]
    game_s = ''
    for k in range(len(all_sets)):
        server = 0 if k==0 else next_server   
        #games = all_sets[k].split(';');length = len(games)
        game_s += get_game_order_sub(all_sets[k] + ';',server) + '.'
        next_server = (server+1)%2 if len(all_sets[k].split(';')) > 12 else (server + len(all_sets[k].split(';')))%2
    return game_s

# takes in s
def get_game_order_sub(s,server):
    games = s.split(';')[:-1]; game_s = ''
    for k in range(len(games)):
        if k==12:
            game_s += str(tbreak_winner(games[k],server))
        else:
            game_s += '0' if server==0 and games[k][-1]=='S' or server==1 and games[k][-1]=='R' else '1'
        server = 1 - server
    return game_s

def tbreak_winner(t_s,server):
    mini_games = t_s.split('/')
    for k in range(1,len(mini_games)):
        server = 1 - server
    return 0 if server==0 and mini_games[-1][-1]=='S' or server==1 and mini_games[-1][-1]=='R' else 1

def predictive_power(col,df):
    # find out how well col does at predicting match winners and losers
    times = 0
    even_indices = []
    for i in range(len(df)):
        if df[col][i][0] > df[col][i][1] and df['winner'][i]==0:
            times += 1
        elif df[col][i][0] < df[col][i][1] and df['winner'][i]==1:
            times += 1
        elif df[col][i][0] == df[col][i][1]:
            even_indices.append(i)
    return times/float(len(df)-len(even_indices)), len(df)-len(even_indices)

# include a pre-match prediction
def do_classify(clf, parameters, indf, featurenames, targetname, target1val, mask, reuse_split=None, score_func=None, n_folds=5, n_jobs=1):
    subdf=indf[featurenames]
    print 'type: ',str(type(clf)).split('.')[-1].split("'")[0]

    Xtrain, Xtest, ytrain, ytest = X[mask], X[~mask], y[mask], y[~mask]
    #print len(Xtrain),len(ytrain)
    clf=clf.fit(Xtrain, ytrain)
    probs_train,probs_test = clf.predict_proba(Xtrain),clf.predict_proba(Xtest)
    train_loss, test_loss = log_loss(ytrain,probs_train,labels=[0,1]),log_loss(ytest,probs_test,labels=[0,1])
    train_accuracy = clf.score(Xtrain, ytrain)
    test_accuracy = clf.score(Xtest, ytest)
    print "############# based on standard predict ################"
    print "Accuracy on training data: %0.2f" % (train_accuracy)
    print "Log Loss on training data: %0.2f" % (train_loss)
    print "Accuracy on test data:     %0.2f" % (test_accuracy)
    print "Log Loss on test data:     %0.2f" % (test_loss)
    return clf, Xtrain, ytrain, Xtest, ytest

def normalize_name(s,tour='atp'):
    if tour=='atp':
        s = s.replace('-',' ')
        s = s.replace('Stanislas','Stan').replace('Stan','Stanislas')
        s = s.replace('Alexandre','Alexander')
        s = s.replace('Federico Delbonis','Federico Del').replace('Federico Del','Federico Delbonis')
        s = s.replace('Mello','Melo')
        s = s.replace('Cedric','Cedrik')
        s = s.replace('Bernakis','Berankis')
        s = s.replace('Hansescu','Hanescu')
        s = s.replace('Teimuraz','Teymuraz')
        s = s.replace('Vikor','Viktor')
        s = s.rstrip()
        s = s.replace('Alex Jr.','Alex Bogomolov')
        s = s.title()
        sep = s.split(' ')
        return ' '.join(sep[:2]) if len(sep)>2 else s
    else:
        return s

def break_point(s):
    s=s.replace('A','S');s=s.replace('D','R')
    all_sets = s.split('.')
    for k in range(len(all_sets)):
        # set server to 0 or 1 at beginning of set, keeping track of all transitions
        server = 0 if k==0 else next_server 
        games = all_sets[k].split(';')
        length = len(games)
        next_server = (server+1)%2 if length>12 else (server + length)%2   
        if k==len(all_sets)-1:
            last_game = games[-1]
            final_server = (server+len(games[:-1]))%2
            pt_s,pt_r = last_game.count('S'),last_game.count('R')
            b_point = pt_r+1>=4 and pt_r+1>=pt_s
            #print b_point
            #print pt_s,pt_r
            if b_point and final_server:
                return (1,0)
            elif b_point and not final_server:
                return (0,1)
            else:
                return (0,0)

# cols is a list of all column sets to test; compare with kls pre-match forecasts
def validate_results(df,columns,n_splits=5):
    kfold = KFold(n_splits=n_splits,shuffle=True)
    scores = np.zeros([len(columns)+2,2,n_splits]);i=0
    for train_ind,test_ind in kfold.split(df):
        lm = linear_model.LogisticRegression(fit_intercept = True)
        train_df,test_df = df.loc[train_ind],df.loc[test_ind]
        
        for k,cols in enumerate(columns):
            lm.fit(train_df[cols].values.reshape([len(train_df),len(cols)]),train_df['winner'])
            y_preds = lm.predict(test_df[cols].values.reshape([len(test_df),len(cols)]))
            y_probs = lm.predict_proba(test_df[cols].values.reshape([len(test_df),len(cols)]))
            scores[k][0][i]=accuracy_score(test_df['winner'],y_preds)
            scores[k][1][i]=log_loss(test_df['winner'],y_probs,labels=[0,1])
        
        y_preds2 = test_df['match_prob_kls']>.5
        y_preds3 = test_df['match_prob_kls_JS']>.5
        scores[len(columns)][0][i]=accuracy_score(test_df['winner'],y_preds2)
        scores[len(columns)][1][i]=log_loss(test_df['winner'],test_df['match_prob_kls'],labels=[0,1])
        scores[len(columns)+1][0][i]=accuracy_score(test_df['winner'],y_preds3)
        scores[len(columns)+1][1][i]=log_loss(test_df['winner'],test_df['match_prob_kls_JS'],labels=[0,1])
        i+=1
    
    for i,cols in enumerate(columns):
        print 'columns: ',cols
        #print '% s_elo used in lm fit: ',lm.coef_[0][1]/(lm.coef_[0][0]+lm.coef_[0][1])
        print 'accuracy: ', np.mean(scores[i][0])
        print 'loss: ', np.mean(scores[i][1])
    
    print 'kls probabilities'
    print 'accuracy: ', np.mean(scores[len(columns)][0])
    print 'loss: ', np.mean(scores[len(columns)][1])

    print 'kls JS probabilities'
    print 'accuracy: ', np.mean(scores[len(columns)+1][0])
    print 'loss: ', np.mean(scores[len(columns)+1][1])

def in_dict(x,d):
    return x in d

# function to cross-validate, with no match-overlap between splits (since there are 100-200
# points per match)
def cross_validate(val_df,clf,cols,target,hyper_parameters,n_splits):
    print 'searching for hyperparams...'
    ids = list(set(val_df['match_id']))
    vfunc = np.vectorize(in_dict)
    kfold = KFold(n_splits=n_splits,shuffle=True)
    key = hyper_parameters.keys()[0]
    scores = [[] for k in range(len(hyper_parameters[key]))]
    
    for train_index,____ in kfold.split(ids):
        train_dict = dict(zip(train_index,[1]*len(train_index)))
        train_ind = vfunc(np.array(val_df['match_id']),train_dict)
        test_ind = (1 - train_ind)==1
        Xtrain, ytrain = val_df[cols][train_ind], val_df[target][train_ind]
        Xtest, ytest = val_df[cols][test_ind], val_df[target][test_ind]
        
        # retrieve classification score for every hyper_parameter fed into this function
        # LOOP THROUGH ALL KEYS here if you want to test multiple hyper_params
        for j in xrange(len(hyper_parameters[key])):
            setattr(clf,key,hyper_parameters[key][j])
            clf.fit(Xtrain,ytrain)
            score = clf.score(Xtest,ytest)
            scores[j].append(score)
    for i in range(len(scores)):
        print hyper_parameters[key][i],': ',np.mean(scores[i])
    best_ind = np.argmax([np.mean(a) for a in scores])
    print 'best: ',{key:hyper_parameters[key][best_ind]}
    return {key:hyper_parameters[key][best_ind]}

if __name__=='__main__':
    S = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R.RRRSSR;RSRRR;SSSS;RSSSS;SSRSS;SRSRSRRSSS;SRSSRS;RRRR;RRSRSSSS.SRRSSRSS;SSSS;RSRSRR;RSRSSS;SSSRS;SSRSS;SSSS;SSSRS;SSSRRRRSR.'
    #S = 'SSSS;RRRR;RSSSS;RSSRRSRSRR;SRRSSS;RRSSRR.SSRRSS;RSSSS;SRRRR;RSRRSR;SRSSRS;SSSRS;SSSS;RRRSSR;'
    S1 = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R;'
    S2 = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R.RRRSSR;RSRRR;SSSS;'
    S3 = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R.RRRSSR;RSRRR;S'
    S4 = 'SS/R.RRRSSR;RSRRR;SSSS;RSSSS;SSRSS;SRSRSRRSSS;SRSSRS;RRRR;RRSRSS'
    a,b = enumerate_pbp(S,'point')









