
import pdb
import math
import random
import pandas as pd
import torch

# function [xks,rst,rstsort,cb,n] = KS_test_time_rescaling(Y,pk)
def timeRescaling(Y, pk, eps=1e-10):
    pk = torch.max(pk, torch.tensor([eps], dtype=torch.double))
    qk = -torch.log(1-pk)
    # make the rescaled times
    n = int(torch.sum(Y).item())
    rISIs = torch.zeros(n-1)
    spikeindicies = Y.nonzero()
    for r in range(n-1):
        total = 0
        ind1 = spikeindicies[r].item()
        ind2 = spikeindicies[r+1].item()
        total = total+torch.sum(qk[ind1+1:ind2-1]).item()
        delta = -(1.0/qk[ind2].item())*math.log(1-random.random()*(1-math.exp(-qk[ind2].item())))
        total = total + qk[ind2].item()*delta
        rISIs[r] = total
    return rISIs

def KSTestTimeRescaling(Y, pk, eps=1e-10):
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
    rISIs = timeRescaling(Y=Y, pk=pk, eps=eps) # rescaled inter-spike intervals
    utRISIs = 1-torch.exp(-rISIs)              # uniform-transformed rescaled inter-spike intervals
    sUTRISIs, indices = torch.sort(utRISIs)             # sorted uniform-transformed rescaled inter-spike intervals
    n = len(rISIs)
    # dt = 1/(n-1)
    # xks = torch.linspace(0.5*dt, 1-0.5*dt, 1.0/dt)
    k = torch.linspace(start=1, end=n, steps=n)
    uCDF = (k-0.5)/n                            # uniform cummulative distribution function
    cb = 1.36/math.sqrt(n)                      # 95% confidence bounds for large sample sizes (see Brown et al., 2001)
    return sUTRISIs, uCDF, cb

def KSTestTimeRescalingUnbinned(spikesTimes, cif, t0, tf, dt, eps=1e-10):
        pk = cif*dt
        bins = torch.arange(t0-dt/2, tf+dt/2, dt)
        # start binning spikes using pandas
        cutRes, _ = pd.cut(spikesTimes, bins=bins, retbins=True)
        Y = torch.from_numpy(cutRes.value_counts().values)
        # end binning spikes using pandas
        sUTRISIs, uCDF, cb = KSTestTimeRescaling(Y=Y, pk=pk, eps=eps)
        return sUTRISIs, uCDF, cb 
