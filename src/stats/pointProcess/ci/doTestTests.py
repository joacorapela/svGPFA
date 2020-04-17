
import sys
import pdb
import math
import random
import numpy as np
from scipy.io import loadmat
import torch
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../../..")
from stats.pointProcess.sampler import Sampler
from stats.pointProcess.tests import KSTestTimeRescalingAnalyticalCorrection, KSTestTimeRescalingNumericalCorrection

def test_compareWilson():
    dataColor = "blue"
    # diagColor = "black"
    cbColor = "red"
    dataLinestyle = "solid"
    # diagLinestyle = "dashed"
    cbLinestyle = "dashed"
    xlabel = "Empirical CDF"
    ylabel = "Model CDF"
    matFilename = "data/dataForKSTestTimeRescalingAnalyticalCorrection.mat"

    loadRes = loadmat(matFilename)
    Y = torch.tensor(loadRes["Y"]).squeeze()
    pk = torch.tensor(loadRes["pk"]).squeeze()
    # empiricalCDF = torch.tensor(loadRes["empiricalCDF"]).squeeze()
    # x = torch.tensor(loadRes["x"]).squeeze()
    # wCB = torch.tensor(loadRes["cb"]).squeeze()
    utSRISIs, uCDF, cb = KSTestTimeRescalingAnalyticalCorrection(Y=Y, pk=pk) # rescaledISIs~exp(\lambda=1.0)
    plt.plot(uCDF, utSRISIs, color=dataColor, linestyle=dataLinestyle)
    # plt.plot([0, 1], [0, 1], color=diagColor, linestyle=diagLinestyle)
    plt.plot([0, 1-cb], [cb, 1], color=cbColor, linestyle=cbLinestyle)
    plt.plot([cb, 1], [0, 1-cb], color=cbColor, linestyle=cbLinestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    pdb.set_trace()

def test_timeRescalingAnalyticalCorrection():
    dt = 1e-3
    t0 = 0.0
    tf = 10.0
    betas = torch.tensor([-3.0, 0.5, -1.0, -.3], dtype=torch.double)
    dataColor = "blue"
    # diagColor = "black"
    cbColor = "red"
    dataLinestyle = "solid"
    # diagLinestyle = "dashed"
    cbLinestyle = "dashed"
    ylabel = "Empirical CDF"
    xlabel = "Model CDF"

    t = torch.arange(t0, tf, dt, dtype=torch.double)
    X = torch.stack((torch.ones(t.shape, dtype=torch.double), 
                     torch.cos(2*math.pi*t), torch.sin(5*math.pi*t), 
                     torch.cos(11*math.pi*t)), 1)
    cifValues = torch.exp(X.matmul(betas))/dt
    sampler = Sampler()
    spikes = torch.tensor(sampler.sampleInhomogeneousPP_timeRescaling(intensityTimes=t, intensityValues=cifValues, T=tf))
    # spikes = torch.tensor(sampler.sampleInhomogeneousPP_thinning(intensityTimes=t, intensityValues=cifValues, T=tf)["inhomogeneous"])
    # spikes = torch.tensor([random.uniform(0, tf) for i in range(len(spikes))])
    bins = np.arange(t0-dt/2, tf+dt/2, dt)
    # start binning spikes using pandas
    cutRes, _ = pd.cut(spikes, bins=bins, retbins=True)
    Y = torch.from_numpy(cutRes.value_counts().values)
    # end binning spikes using pandas
    Y = Y[torch.randperm(Y.shape[0])]
    pk = cifValues*dt
    # pk = cifValues[torch.randperm(cifValues.shape[0])]*dt
    utSRISIs, uCDF, cb = KSTestTimeRescaling(Y=Y, pk=pk) # rescaledISIs~exp(\lambda=1.0)

    plt.subplot(2, 1, 1)
    plt.plot(t, pk)

    plt.subplot(2, 1, 2)
    plt.plot(utSRISIs, uCDF, color=dataColor, linestyle=dataLinestyle)
    # plt.plot([0, 1], [0, 1], color=diagColor, linestyle=diagLinestyle)
    plt.plot([0, 1-cb], [cb, 1], color=cbColor, linestyle=cbLinestyle)
    plt.plot([cb, 1], [0, 1-cb], color=cbColor, linestyle=cbLinestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Number of spikes {:d}".format(Y.sum()))

    plt.show()

    pdb.set_trace()

def test_timeRescalingNumericalCorrection():
    dt = 1e-3
    t0 = 0.0
    tf = 10.0
    gamma = 3
    betas = torch.tensor([-3.0, 0.5, -1.0, -.3], dtype=torch.double)

    refColor = "black"
    dataColor = "blue"
    cbColor = "red"
    refLinestyle = "solid"
    dataLinestyle = "solid"
    cbLinestyle = "dashed"
    dataMarker = "*"
    ylabel = "CDF Difference"
    xlabel = "Rescaled Time"

    t = torch.arange(t0, tf, dt, dtype=torch.double)
    X = torch.stack((torch.ones(t.shape, dtype=torch.double),
                     torch.cos(2*math.pi*t), torch.sin(5*math.pi*t),
                     torch.cos(11*math.pi*t)), 1)
    cifValues = torch.exp(X.matmul(betas))/dt
    sampler = Sampler()
    spikes = torch.tensor(sampler.sampleInhomogeneousPP_timeRescaling(intensityTimes=t, intensityValues=cifValues, T=tf))
    # spikes = torch.tensor(sampler.sampleInhomogeneousPP_thinning(intensityTimes=t, intensityValues=cifValues, T=tf)["inhomogeneous"])
    # spikes = torch.tensor([random.uniform(0, tf) for i in range(len(spikes))])
    diffECDFsX, diffECDFsY, cb = KSTestTimeRescalingNumericalCorrection(spikesTimes=spikes, cifTimes=t, cifValues=cifValues, gamma=gamma)

    plt.plot(diffECDFsX, diffECDFsY, color=dataColor, marker=dataMarker, linestyle=dataLinestyle)
    plt.axhline(0, color=refColor, linestyle=refLinestyle)
    plt.axhline(cb, color=cbColor, linestyle=cbLinestyle)
    plt.axhline(-cb, color=cbColor, linestyle=cbLinestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Number of spikes {:d}".format(len(spikes)))

    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    # test_timeRescaling()
    test_compareWilson()
    # test_timeRescalingNumericalCorrection()
