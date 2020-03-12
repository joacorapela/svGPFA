import math
import random
import torch

# function [xks,rst,rstsort,cb,n] = KS_test_time_rescaling(Y,pk)
def timeRescaling(Y, pk, eps=1e-10):
    '''
    Y: binary sequence for the discrete time point process
    pk: event probability in each time been based on a conditional intensity function

    xks: uniform CDF
    rst: time rescaled times
    cb: 95% confidence bounds
    n: number of spikes

    KS-test based on time-rescalling theorem =====================
    Use DT Correction for Time Rescaling Theorem - Haslinger, Pipa and Brown (2010)
    discrete time conditional probabilities "pk"  where 0<=pk<= 1
    '''
    pk = max(pk, eps)
    qk = -torch.log(1-pk)
    # make the rescaled times
    n = torch.sum(Y)
    rst = torch.zeros(n-1, 1)
    spikeindicies = Y.nonzero()
    for r in range(n-1):
        total = 0
        ind1 = spikeindicies[r]
        ind2 = spikeindicies(r+1)
        total = total+torch.sum(qk[ind1+1:ind2-1])
        delta = -(1.0/qk(ind2))*math.log(1-random.random()*(1-math.exp(-qk[ind2])))
        total = total + qk[ind2]*delta
        rst[r] = total
    # inrst=1/(n-1);
    # xks=(0.5*inrst:inrst:1-0.5*inrst)';
    # cb=1.36*sqrt(inrst)
    return rst
