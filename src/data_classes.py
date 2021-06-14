import numpy as np
from collections import defaultdict

'''
tracking object for player's year-long performance over time
accepts dates in (year,month)
last_year contains last 12 months stats, most recent to least
'''
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
            self.last_year[diff:] = self.last_year[:12-diff]; self.last_year[:diff] = 0 ## Doing this is dangerous!!!
        self.most_recent = match_date

    def update(self,match_date,match_stats):
        self.set_month(match_date)
        self.last_year[0] = self.last_year[0]+match_stats

'''
tracking object for opponent-adjusted ratings
stores opponent ability at time of match to compare performance against
'''
class adj_stats_52():
    def __init__(self,date):
        self.most_recent = date
        self.last_year = np.zeros([12,6])
        self.adj_sr = [0,0]

    def time_diff(self,new_date,old_date):
        return 12*(new_date[0]-old_date[0])+(new_date[1]-old_date[1])

    def set_month(self,match_date):
        diff = self.time_diff(match_date,self.most_recent)
        if diff>=12:
            self.last_year = np.zeros([12,6])
        elif diff>0:
            self.last_year[diff:] = self.last_year[:12-diff]; self.last_year[:diff] = 0
        self.most_recent = match_date
        self.update_adj_sr()

    def update(self,match_date,match_stats):
        self.set_month(match_date)
        self.last_year[0] = self.last_year[0]+match_stats
        self.update_adj_sr()

    # update the player's adjust serve/return ability, based on last twelve months
    def update_adj_sr(self):
        s_pt, r_pt = np.sum(self.last_year[:,1]), np.sum(self.last_year[:,3])
        if s_pt==0 or r_pt==0:
            self.adj_sr = [0,0]
            return
        with np.errstate(divide='ignore', invalid='ignore'):
            f_i = np.sum(self.last_year[:,0])/s_pt
            f_adj = 1 - np.sum(self.last_year[:,4])/s_pt
            g_i = np.sum(self.last_year[:,2])/r_pt
            g_adj = 1 - np.sum(self.last_year[:,5])/r_pt
        self.adj_sr[0] = f_i - f_adj
        self.adj_sr[1] = g_i - g_adj

'''
tracking object for common-opponent ratings
stores all historical performance against opponents
'''
class commop_stats():
    def __init__(self):
        self.history = defaultdict(lambda: np.zeros(4))

    def update(self, match_stats, opponent_name):
        self.history[opponent_name] += match_stats

'''
tracking object for common-opponent ratings
stores past year of performance against opponents
'''
# class commop_stats_52():
#     def __init__(self, date):
#         self.last_year = defaultdict(lambda: np.zeros([12, 4]))
#         self.most_recent = date

#     def time_diff(self, new_date, old_date):
#         return 12*(new_date[0]-old_date[0])+(new_date[1]-old_date[1])

#     def update_player_stats(self, match_date, opponent_name):
#         diff = self.time_diff(match_date, self.most_recent)
#         if diff>=12:
#             self.last_year[opponent_name] = np.zeros([12,4])
#         elif diff>0:
#             self.last_year[opponent_name][diff:] = self.last_year[opponent_name][:12-diff]
#             self.last_year[opponent_name][:diff] = 0

#     def update_player_histories(self, match_date, opponent_name):
#         for opp_name in np.union1d(opponent_name, self.last_year.keys()):
#             self.update_player_stats(match_date, opp_name)

#         self.most_recent = match_date

#     def update(self, match_date, match_stats, opponent_name):
#         self.update_player_histories(match_date, opponent_name)
#         self.last_year[opponent_name][0] = self.last_year[opponent_name][0]+match_stats

'''
tracking object for yearly tournament averages
'''
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
