# -*- coding: utf-8 -*-
"""
    elo
    ~~~

    The Elo rating system.

    :copyright: (c) 2012 by Heungsub Lee
    :license: BSD, see LICENSE for more details.
"""
import math
import inspect
#: Default K-factor.
#K_FACTOR = 10
#: Default rating class.
RATING_CLASS = float
#: Default initial rating.
INITIAL = 1500
#: Default Beta value.
BETA = 200

WIN = 1.
LOSS = 0.


class Rating(object):
    def __init__(self, value=1500,times=1):
        self.value = value
        self.times = times

    def __repr__(self):
        c = type(self)
        ext_params = inspect.getargspec(c.__init__)[0][2:]
        kwargs = ', '.join('%s=%r' % (param, getattr(self, param))
                           for param in ext_params)
        if kwargs:
            kwargs = ', ' + kwargs
        args = ('.'.join([c.__module__, c.__name__]), self.value, kwargs)
        return '%s(%.3f%s)' % args


class Elo_Rater(object):
    def __init__(self, rating_class=RATING_CLASS,
                 initial=INITIAL, beta=BETA):
        self.rating_class = rating_class
        self.initial = initial
        self.beta = beta

    def expect(self, rating, other_rating_val):
        """The "E" function in Elo. It calculates the expected score of the
        first rating by the second rating.
        """
        # http://www.chess-mind.com/en/elo-system
        diff = float(other_rating_val) - float(rating.value)
        f_factor = 2 * self.beta  # rating disparity
        return 1. / (1 + 10 ** (diff / f_factor))

    def adjust(self, rating, series):
        """Calculates the adjustment value."""
        return series[0] - self.expect(rating, series[1])

    def step_decay_k(self, rating, counts):
        if counts:
            initial_lrate = 0.1
            drop = 0.05
            epochs_drop = 10000.0
            lrate = initial_lrate * math.pow(drop,  
                    math.floor((1+rating.times)/epochs_drop))
            return lrate
        else:
            return 32

    def exp_decay_k(self, rating, counts):
        if counts:
            initial_lrate = 0.1
            k = 0.01
            return initial_lrate * math.exp(-k*rating.times)
        else:
            return 32

    def calculate_k(self,rating,counts):
        #return max(250/((rating.times+5)**.4),32)
        if counts:
            return 250/((rating.times+5)**.4)
        else:
            return 32
    
    def s_tournament(self, tny_name):
#         if "Grand Slam" in tny_name:
#             tny_scale = 1.0
#         elif "Tour Finals" in tny_name:
#             tny_scale = 0.9
#         elif "Masters" in tny_name:
#             tny_scale = 0.85
#         elif "Olympics" in tny_name:
#             tny_scale = 0.8
#         else:
#             tny_scale = 0.7
        if "Australian Open" in tny_name:
            tny_scale = 1.0
        if "US Open" in tny_name:
            tny_scale = 1.0
        if "Wimbledon" in tny_name:
            tny_scale = 1.0
        if "Roland Garros" in tny_name:
            tny_scale = 1.0
        elif "Finals" in tny_name:
            tny_scale = 0.9
        elif "Masters" in tny_name:
            tny_scale = 0.85
        elif "Olympics" in tny_name:
            tny_scale = 0.8
        else:
            tny_scale = 0.7
        return tny_scale
    
    def s_tny_round_name(self, tny_round_name):
        if "Finals" in tny_round_name:
            tny_round_name_scale = 1.0
        if "Semi-Finals" in tny_round_name:
            tny_round_name_scale = 0.9
        if "Quarter-Finals" in tny_round_name:
            tny_round_name_scale = 0.85
        if "Round of 16" in tny_round_name:
            tny_round_name_scale = 0.8
        elif "Round of 32" in tny_round_name:
            tny_round_name_scale = 0.8
        elif "Round of 64" in tny_round_name:
            tny_round_name_scale = 0.75
        elif "Round of 128" in tny_round_name:
            tny_round_name_scale = 0.75
        elif "Bronze" in tny_round_name:
            tny_round_name_scale = 0.95
        else:
            tny_round_name_scale = 0.7
        return tny_round_name_scale
    
    def avg_scalers(self, scalers):
        return sum(scalers)/len(scalers)

    def rate(self, rating, series, is_gs=False, counts=False, tny_name="", tny_round_name=""):
        """Calculates new ratings by the game result series."""
        k = self.calculate_k(rating,counts)*1.1 if is_gs else self.calculate_k(rating,counts)
        # k = self.exp_decay_k(rating,counts)*1.1 if is_gs else self.exp_decay_k(rating,counts)
        # k = self.step_decay_k(rating,counts)*1.1 if is_gs else self.step_decay_k(rating,counts)

        # original
        rating.value = float(rating.value) + k * self.adjust(rating, series)

        # calculate scaler base on current rating
        # rate_scale = 1+18/(1+2**((float(rating.value)-1500)/63))

        # calculate scaler base on tournament level
        # tny_scale = self.s_tournament(tny_name)

        # calculate scaler base on match type
        # tny_round_scale = self.s_tournament(tny_round_name)

        # rating.value = float(rating.value) + k * self.adjust(rating, series) * self.avg_scalers([tny_round_scale,tny_scale,rate_scale])
        # rating.value = float(rating.value) + k * self.adjust(rating, series) * tny_round_scale

        rating.times += 1
        return rating

    # def adjust_1vs1(self, rating1, rating2, drawn=False):
    #     return self.adjust(rating1, [(DRAW if drawn else WIN, rating2)])

    def rate_1vs1(self, rating1, rating2, is_gs=False,counts=True, tny_name="", tny_round_name=""):
        scores = (WIN, LOSS)
        r1,r2 = rating1.value, rating2.value
        return (self.rate(rating1, [scores[0], r2], is_gs, counts, tny_name, tny_round_name),
                self.rate(rating2, [scores[1], r1], is_gs, counts, tny_name, tny_round_name))

