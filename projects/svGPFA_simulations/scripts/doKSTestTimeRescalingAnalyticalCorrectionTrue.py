import sys
import os
import pdb
import math
import torch
import pickle
import statsmodels.tsa.stattools
sys.path.append("../src")
import plot.svGPFA.plotUtils
from stats.pointProcess.tests import KSTestTimeRescalingAnalyticalCorrectionUnbinned
def main(argv):
    if len(argv)!=5:
        print("Usage {:s} <simulation result number> <number of KS test resamples> <trial> <neuron>".format(argv[0]))
        return

    simResNumber = int(argv[1])
    gamma = int(argv[2])
    trialToAnalyze = int(argv[3])
    neuronToAnalyze = int(argv[4])
    dtCIF = 1e-3

    simResFilename = "results/{:08d}_simRes.pickle".format(simResNumber)
    ksTestTimeRescalingFigFilename = "figures/{:08d}_simulation_ksTestTimeRescaling_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(simResNumber, trialToAnalyze, neuronToAnalyze)
    timeRescalingDiffCDFsFigFilename = "figures/{:08d}_simulation_timeRescalingDiffCDFs_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(simResNumber, trialToAnalyze, neuronToAnalyze)
    timeRescaling1LagScatterPlotFigFilename = "figures/{:08d}_simulation_timeRescaling1LagScatterPlot_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(simResNumber, trialToAnalyze, neuronToAnalyze)
    timeRescalingACFFigFilename = "figures/{:08d}_simulation_timeRescalingACF_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(simResNumber, trialToAnalyze, neuronToAnalyze)


    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    cifTimes = simRes["cifTimes"]
    cifValues = simRes["cifValues"]

    spikesTimesKS = spikesTimes[trialToAnalyze][neuronToAnalyze]
    cifTimesKS = cifTimes[trialToAnalyze]
    cifValuesKS = cifValues[trialToAnalyze][neuronToAnalyze]
    t0 = math.floor(cifTimesKS.min())
    tf = math.ceil(cifTimesKS.max())
    dt = (cifTimesKS[1]-cifTimesKS[0]).item()

    utSRISIs, uCDF, cb, utRISIs = KSTestTimeRescalingAnalyticalCorrectionUnbinned(spikesTimes=spikesTimesKS, cifValues=cifValuesKS, t0=t0, tf=tf, dt=dt)
    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(trialToAnalyze, neuronToAnalyze, len(spikesTimesKS))
    sUTRISIs, _ = torch.sort(utSRISIs)
    plot.svGPFA.plotUtils.plotResKSTestTimeRescalingAnalyticalCorrection(sUTRISIs=sUTRISIs, uCDF=uCDF, cb=cb, title=title, figFilename=ksTestTimeRescalingFigFilename)
    plot.svGPFA.plotUtils.plotDifferenceCDFs(sUTRISIs=sUTRISIs, uCDF=uCDF, cb=cb, title=title, figFilename=timeRescalingDiffCDFsFigFilename)
    plot.svGPFA.plotUtils.plotScatter1Lag(x=utRISIs, title=title, figFilename=timeRescaling1LagScatterPlotFigFilename),
    acfRes, confint = statsmodels.tsa.stattools.acf(x=utRISIs, unbiased=True, alpha=0.05)
    plot.svGPFA.plotUtils.plotACF(acf=acfRes, Fs=1/dt, confint=confint, title=title, figFilename=timeRescalingACFFigFilename),

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
