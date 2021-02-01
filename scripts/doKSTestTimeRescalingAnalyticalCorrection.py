import sys
import os
import pdb
import math
import torch
import pickle
import configparser
import statsmodels.tsa.stattools
sys.path.append("../src")
import plot.svGPFA.plotUtils
from stats.pointProcess.tests import KSTestTimeRescalingAnalyticalCorrectionUnbinned

def main(argv):
    if len(argv)<4:
        print("Usage {:s} <estimation result number> <trial to plot> <neuron 1 to plot> <neuron 2 to plot> ... ".format(argv[0]))
        return

    estResNumber = int(argv[1])
    trialToPlot = int(argv[2])
    dtCIF = 1e-3

    # load data and initial values
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)
    with open(modelSaveFilename, "rb") as f: savedResults = pickle.load(f)
    model = savedResults["model"]

    estResConfigFilename = "results/{:08d}_estimation_metaData.ini".format(estResNumber)
    estResConfig = configparser.ConfigParser()
    estResConfig.read(estResConfigFilename)
    simResNumber = int(estResConfig["simulation_params"]["simResNumber"])
    simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)

    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]
    simResFilename = simResConfig["simulation_results"]["simResFilename"]

    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nLatents = int(simInitConfig["control_variables"]["nLatents"])
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]

    # KS test time rescaling
    T = torch.tensor(trialsLengths).max()
    oneTrialCIFTimes = torch.arange(0, T, dtCIF)
    cifTimes = torch.unsqueeze(torch.ger(torch.ones(nTrials), oneTrialCIFTimes), dim=2)
    with torch.no_grad():
        # cifs = model.sampleCIFs(times=cifTimes)
        cifsValues = model.computeMeanCIFs(times=cifTimes)

    for i in range(3, len(argv)):
        print("Processing neuron {:d}".format(i))
        neuronToPlot = int(argv[i])
        ksTestTimeRescalingFigFilename = "figures/{:08d}_ksTestTimeRescaling_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(estResNumber, trialToPlot, neuronToPlot)
        timeRescalingDiffCDFsFigFilename = "figures/{:08d}_timeRescalingDiffCDFs_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(estResNumber, trialToPlot, neuronToPlot)
        timeRescaling1LagScatterPlotFigFilename = "figures/{:08d}_timeRescaling1LagScatterPlot_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(estResNumber, trialToPlot, neuronToPlot)
        timeRescalingACFFigFilename = "figures/{:08d}_timeRescalingACF_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(estResNumber, trialToPlot, neuronToPlot)

        spikesTimesKS = spikesTimes[trialToPlot][neuronToPlot]
        ### begin debug ###
        # print("*** Warning debug code on ***")
        # import random
        # spikesTimesKS = [random.uniform(0, 1) for i in range(len(spikesTimesKS))]
        ### end debug ###
        cifTimesKS = cifTimes[trialToPlot,:,0]
        cifValuesKS = cifsValues[trialToPlot][neuronToPlot]
        t0 = math.floor(cifTimesKS.min())
        tf = math.ceil(cifTimesKS.max())
        dt = (cifTimesKS[1]-cifTimesKS[0]).item()
        utSRISIs, uCDF, cb, utRISIs = KSTestTimeRescalingAnalyticalCorrectionUnbinned(spikesTimes=spikesTimesKS, cifValues=cifValuesKS, t0=t0, tf=tf, dt=dt)
        title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(trialToPlot, neuronToPlot, len(spikesTimesKS))
        sUTRISIs, _ = torch.sort(utSRISIs)
        plot.svGPFA.plotUtils.plotResKSTestTimeRescalingAnalyticalCorrection(sUTRISIs=sUTRISIs, uCDF=uCDF, cb=cb, title=title, figFilename=ksTestTimeRescalingFigFilename)
        plot.svGPFA.plotUtils.plotDifferenceCDFs(sUTRISIs=sUTRISIs, uCDF=uCDF, cb=cb, figFilename=timeRescalingDiffCDFsFigFilename),
        plot.svGPFA.plotUtils.plotScatter1Lag(x=utRISIs, title=title, figFilename=timeRescaling1LagScatterPlotFigFilename)
        acfRes, confint = statsmodels.tsa.stattools.acf(x=utRISIs, unbiased=True, alpha=0.05)
        plot.svGPFA.plotUtils.plotACF(acf=acfRes, Fs=1/dt, confint=confint, title=title, figFilename=timeRescalingACFFigFilename),

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
