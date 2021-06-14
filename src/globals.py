from datetime import timedelta, datetime as dt

DATE = (dt.now() + timedelta(days=2)).strftime(('%m_%d_%Y'))
CURRENT_YEAR = int(DATE.split('_')[-1])
TOUR = 'atp'
START_YEAR = 1968
COMMOP_START_YEAR = 2010
RET_STRINGS = (
    'ABN', 'DEF', 'In Progress', 'RET', 'W/O', ' RET', ' W/O', 'nan', 'walkover'
)
ABD_STRINGS = (
    'abandoned', 'ABN', 'ABD', 'DEF', 'def', 'unfinished', 'Walkover'
)
COUNTS_538 = 1
PBP_COLS = [
    'sets_0','sets_1','games_0','games_1',
    'points_0','points_1','tp_0','tp_1',
    'p0_swp','p0_sp','p1_swp','p1_sp',
    'server'
]
# EPSILON = .001
EPSILON = .00000000001
