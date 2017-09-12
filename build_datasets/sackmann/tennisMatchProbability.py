## calculates probability of winning a tennis match from any given score dependent on the skill levels
## of the two players
 
from tennisGameProbability import gameProb
from tennisSetProbability import setGeneral
from tennisTiebreakProbability import tiebreakProb
 
def fact(x):
    if x in [0, 1]:  return 1
    r = 1
    for a in range(1, (x+1)):  r = r*a
    return r
 
def ch(a, b):
    return fact(a)/(fact(b)*fact(a-b))
 
def matchGeneral(e, v=0, w=0, s=3):
    ## calculates probability of winning the match
    ## from the beginning of a set
    ## e is p(winning a set)
    ## v and w is current set score
    ## s is total number of sets ("best of")
    towin = (s+1)/2
    left = towin - v
    if left == 0:   return 1
    remain = s - v - w
    if left > remain:   return 0
    win = 0
    #print left,remain
    for i in range(left, (remain+1)):
        # print ch(i-1, left-1)
        # print 'e, left-1',e,left-1
        # print type(e)
        # print (e**(left-1))
        # print ((1-e)**(i-left))
        add = ch(i-1, left-1)*(e**(left-1))*((1-e)**(i-left))*e
        win += add
    return win
 
def matchProb(s, t, gv=0, gw=0, sv=0, sw=0, mv=0, mw=0, sets=3):
    ## calculates probability of winning a match from any given score,
    ## given:
    ## s, t: p(server wins a service point), p(server wins return point)
    ## gv, gw: current score within the game. e.g. 30-15 is 2, 1
    ## sv, sw: current score within the set. e.g. 5, 4
    ## mv, mw: current score within the match (number of sets for each player)
    ## v's are serving player; w's are returning player
    ## sets: "best of", so default is best of 3
    a = gameProb(s)
    b = gameProb(t)
    c = setGeneral(s, t)
    #print 'set prob: ',c[0]
    if gv == 0 and gw == 0: ## no point score
        if sv == 0 and sw == 0: ## no game score
            #print 'mwin: ', matchGeneral(c[0], v=mv, w=mw, s=sets)
            return matchGeneral(c[0], v=mv, w=mw, s=sets)
        else:   ## we're in mid-set, no point score
            sWin = setGeneral(s, t, v=sv, w=sw)[0]
            sLoss = 1 - sWin
    elif sv == 6 and sw == 6:         
        sWin = tiebreakProb(s, t, v=gv, w=gw)
        sLoss = 1 - sWin       
    else:
        gWin = gameProb(s, v=gv, w=gw)
        gLoss = 1 - gWin
        sWin = gWin*(1 - setGeneral(1-t, 1-s, v=sw, w=sv+1)[0])
        sWin += gLoss*(1 - setGeneral(1-t, 1-s, v=sw+1, w=sv)[0])
        sLoss = 1 - sWin
    mWin = sWin*matchGeneral(c[0], v=mv+1, w=mw, s=sets)
    mWin += sLoss*matchGeneral(c[0], v=mv, w=mw+1, s=sets)
    #print 'mwin: ', mWin
    return mWin

if __name__=='__main__':
    # diff = .1
    # for i in range(10):
    #     print 'p_s =',i*.1+.1
    #     print matchProb(i*.1+.1,1-i*.1)
    print matchProb(.5923,1-.5109,sets=5)
    # print matchProb(.5,.4,sets=3)
    # print matchProb(.6481,.8484,sets=3)
    for diff in [.01,.02,.03,.04,.05]:
        print matchProb(.64+diff,1-.64+diff)

    #print matchProb(.66,.35)

