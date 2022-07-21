
import pdb
import math
import warnings
import random
import pandas as pd
import numpy as np
import torch

from . import sampling

def timeRescaling(Y, pk):
    n = int(torch.sum(Y).item())
    if n==0:
        return None
    rISIs = torch.zeros(n-1, dtype=torch.double)
    spikeindicies = Y.nonzero()
    for r in range(n-1):
        ind1 = spikeindicies[r].item()
        ind2 = spikeindicies[r+1].item()
        rISIs[r] = torch.sum(pk[ind1+1:ind2+1]).item()
    return rISIs

def timeRescalingForUnbinnedSpikes(spikesTimes, cifValues, t0, tf, dt):
    pk = cifValues*dt
    bins = torch.arange(t0-dt/2, tf+dt/2, dt)
    # start binning spikes using pandas
    cutRes, _ = pd.cut(spikesTimes, bins=bins, retbins=True)
    Y = torch.from_numpy(cutRes.value_counts().values)
    # end binning spikes using pandas
    indicesMoreThanOneSpikes = (Y>1).nonzero()
    if len(indicesMoreThanOneSpikes)>0:
        warnings.warn("Found more than one spike in {:d} bins".format(len(indicesMoreThanOneSpikes)))
        Y[indicesMoreThanOneSpikes] = 1.0
    zExp = timeRescaling(Y=Y, pk=pk)
    return zExp

def KSTestTimeRescalingNumericalCorrection(spikesTimes, cifTimes, cifValues, gamma):
    # from Haslinger et al., 2010, p. 2492,
    # Procedure for Numerical Correction

    def subtractECDFs(ecdf1X, ecdf1Y, ecdf2X, ecdf2Y):
        subX = torch.sort(torch.cat((ecdf1X, ecdf2X)))[0].numpy()
        ecdf1YI = np.interp(subX, ecdf1X.numpy(), ecdf1Y.numpy())
        ecdf2YI = np.interp(subX, ecdf2X.numpy(), ecdf2Y.numpy())
        subY = ecdf1YI-ecdf2YI
        # begin debug
        # plt.plot(subX, ecdf1YI)
        # plt.plot(subX, ecdf2YI)
        # plt.show()
        # pdb.set_trace()
        # end debug
        return torch.tensor(subX), torch.tensor(subY)

    t0 = cifTimes.min()
    tf = cifTimes.max()
    dt = cifTimes[1]-cifTimes[0]
    T = tf-t0
    print("Processing given ISIs")
    zExp = timeRescalingForUnbinnedSpikes(spikesTimes=spikesTimes, cifValues=cifValues, t0=t0, tf=tf, dt=dt)
    zSim = None
    for i in range(gamma):
        print("Processing iter {:d}/{:d}".format(i, gamma-1))
        simSpikesTimes = sampling.sampleInhomogeneousPP_timeRescaling(CIF_times=cifTimes, CIF_values=cifValues)
        zSimIter = timeRescalingForUnbinnedSpikes(spikesTimes=simSpikesTimes,
                                                  cifValues=cifValues,
                                                  t0=t0, tf=tf, dt=dt)
        if zSimIter is not None:
            if zSim is None:
                zSim = zSimIter
            else:
                zSim = torch.cat((zSim, zSimIter))
    expECDFx, _ = torch.sort(zExp)
    expECDFy = torch.linspace(0, 1, len(zExp))
    simECDFx, _ = torch.sort(zSim)
    simECDFy = torch.linspace(0, 1, len(zSim))
    diffECDFsX, diffECDFsY = subtractECDFs(ecdf1X=expECDFx, ecdf1Y=expECDFy, ecdf2X=simECDFx, ecdf2Y=simECDFy)

    Nexp = len(zExp)
    Nsim = len(zSim)
    cb = 1.36*math.sqrt((Nexp+Nsim)/(Nexp*Nsim))

    return diffECDFsX, diffECDFsY, expECDFx, expECDFy, simECDFx, simECDFy, cb

def timeRescalingAnalyticalCorrection(Y, pk, eps=1e-10):
    indicesMoreThanOneSpikes = (Y>1).nonzero()
    if len(indicesMoreThanOneSpikes)>0:
        warnings.warn("Found more than one spike in {:d} bins".format(len(indicesMoreThanOneSpikes)))
        Y[indicesMoreThanOneSpikes] = 1.0
    pk = torch.max(pk, torch.tensor([eps], dtype=torch.double))
    qk = -torch.log(1-pk)
    # make the rescaled times
    n = int(torch.sum(Y).item())
    rISIs = torch.zeros(n-1)
    spikeindicies = Y.nonzero()
    for r in range(n-1):
        ind1 = spikeindicies[r].item()
        ind2 = spikeindicies[r+1].item()
        aSum = torch.sum(qk[(ind1+1):ind2]).item()
        # delta = -(1.0/qk[ind2].item())*math.log(1-random.random()*(1-math.exp(-qk[ind2].item())))
        # total = total + qk[ind2].item()*delta
        delta = -math.log(1-random.random()*(1-math.exp(-qk[ind2].item())))
        rISIs[r] = aSum + delta
    return rISIs

def KSTestTimeRescalingAnalyticalCorrection(Y, pk, eps=1e-10):
    '''
    Y: binary sequence for the discrete time point process
    pk: event probability in each time been based on a conditional cif function

    xks: uniform CDF
    rst: time rescaled times
    cb: 95% confidence bounds
    n: number of spikes

    KS-test based on time-rescalling theorem =====================
    Use DT Correction for Time Rescaling Theorem - Haslinger, Pipa and Brown (2010)
    discrete time conditional probabilities "pk"  where 0<=pk<= 1
    '''
    rISIs = timeRescalingAnalyticalCorrection(Y=Y, pk=pk, eps=eps) # rescaled inter-spike intervals
    utRISIs = 1-torch.exp(-rISIs)              # uniform-transformed srescaled inter-spike intervals
    srISIs, _ = torch.sort(rISIs)                 # sorted rescaled inter-spike intervals
    utSRISIs = 1-torch.exp(-srISIs)              # uniform-transformed sorted rescaled inter-spike intervals
    n = len(rISIs)
    # dt = 1/(n-1)
    # xks = torch.linspace(0.5*dt, 1-0.5*dt, 1.0/dt)
    k = torch.linspace(start=1, end=n, steps=n)
    uCDF = (k-0.5)/n                            # uniform cummulative distribution function
    cb = 1.36/math.sqrt(n)                      # 95% confidence bounds for large sample sizes (see Brown et al., 2001)
    return utSRISIs, uCDF, cb, utRISIs

def KSTestTimeRescalingAnalyticalCorrectionUnbinned(spikesTimes, cifValues, t0, tf, dt, eps=1e-10):
        pk = cifValues*dt
        bins = torch.arange(t0-dt/2, tf+dt/2, dt)
        # start binning spikes using pandas
        cutRes, _ = pd.cut(spikesTimes, bins=bins, retbins=True)
        Y = torch.from_numpy(cutRes.value_counts().values)
        # end binning spikes using pandas
        utSRISIs, uCDF, cb, utRISIs = KSTestTimeRescalingAnalyticalCorrection(Y=Y, pk=pk, eps=eps)
        return utSRISIs, uCDF, cb, utRISIs
