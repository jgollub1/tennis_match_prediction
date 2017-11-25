# -*- coding: utf-8 -*-
"""
    elo
    ~~~

    The Elo rating system.

    :copyright: (c) 2012 by Heungsub Lee
    :license: BSD, see LICENSE for more details.
"""

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

    def calculate_k(self,rating,counts):
        #return max(250/((rating.times+5)**.4),32)
        if counts:
            return 250/((rating.times+5)**.4)
        else:
            return 32

    def rate(self, rating, series, is_gs=False, counts=False):
        """Calculates new ratings by the game result series."""
        k = self.calculate_k(rating,counts)*1.1 if is_gs else self.calculate_k(rating,counts)
        rating.value = float(rating.value) + k * self.adjust(rating, series)
        rating.times += 1
        return rating

    # def adjust_1vs1(self, rating1, rating2, drawn=False):
    #     return self.adjust(rating1, [(DRAW if drawn else WIN, rating2)])

    def rate_1vs1(self, rating1, rating2, is_gs=False,counts=True):
        scores = (WIN, LOSS)
        r1,r2 = rating1.value, rating2.value
        return (self.rate(rating1, [scores[0], r2],is_gs,counts),
                self.rate(rating2, [scores[1], r1],is_gs,counts))

