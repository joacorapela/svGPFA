
import sys
import pdb
import math
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../..")
from stats.pointProcess.sampler import Sampler
from stats.pointProcess.tests import KSTestTimeRescaling


def test_timeRescaling():
    dt = 1e-3
    t0 = 0.0
    tf = 20.0
    baselineRate = 20
    modulationAmplitudeRate = 2
    dataColor = "blue"
    # diagColor = "black"
    cbColor = "red"
    dataLinestyle = "solid"
    # diagLinestyle = "dashed"
    cbLinestyle = "dashed"
    ylabel = "Empirical CDF"
    xlabel = "Model CDF"

    cif = lambda t: baselineRate+modulationAmplitudeRate*torch.sin(2*math.pi*t)
    t = torch.arange(t0, tf, dt)
    cifValues = cif(t=t)
    sampler = Sampler()
    spikes = torch.tensor(sampler.sampleInhomogeneousPP_timeRescaling(intensityTimes=t, intensityValues=cifValues, T=tf))
    # spikes = torch.tensor(sampler.sampleInhomogeneousPP_thinning(intensityTimes=t, intensityValues=cifValues, T=tf)["inhomogeneous"])
    bins = np.arange(t0-dt/2, tf+dt/2, dt)
    # start binning spikes using pandas
    cutRes, _ = pd.cut(spikes, bins=bins, retbins=True)
    Y = torch.from_numpy(cutRes.value_counts().values)
    # end binning spikes using pandas
    pk = cifValues*dt
    sUTRISIs, uCDF, cb = KSTestTimeRescaling(Y=Y, pk=pk) # rescaledISIs~exp(\lambda=1.0)

    plt.plot(sUTRISIs, uCDF, color=dataColor, linestyle=dataLinestyle)
    # plt.plot([0, 1], [0, 1], color=diagColor, linestyle=diagLinestyle)
    plt.plot([0, 1-cb], [cb, 1], color=cbColor, linestyle=cbLinestyle)
    plt.plot([cb, 1], [0, 1-cb], color=cbColor, linestyle=cbLinestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    test_timeRescaling()
