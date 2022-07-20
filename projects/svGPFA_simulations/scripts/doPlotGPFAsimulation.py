
import pdb
import sys
import os
import random
import torch
import plotly
import pickle
import argparse
import configparser
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
sys.path.append("../src")
import plot.svGPFA.plotUtils
import plot.svGPFA.plotUtilsPlotly
import stats.pointProcess.tests
import utils.svGPFA.miscUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simResNumber", help="Simulation result number", type=int)
    parser.add_argument("--exampleTrial", help="Trial to plot example traces", type=int, default=0)
    parser.add_argument("--exampleNeuron", help="Neuron to plot example traces", type=int, default=0)
    parser.add_argument("--nResamplesKSTest", help="Number of resamples for KS test", type=int, default=10)
    args = parser.parse_args()

    simResNumber = args.simResNumber
    exampleTrial = args.exampleTrial
    exampleNeuron = args.exampleNeuron
    nResamplesKSTest = args.nResamplesKSTest

    simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]

    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nLatents = int(simInitConfig["control_variables"]["nLatents"])
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    dtCIF = float(simInitConfig["control_variables"]["dtCIF"])
    nTrials = len(trialsLengths)
    T = torch.tensor(trialsLengths).max().item()

    simResFilename = "results/{:08d}_simRes.pickle".format(simResNumber)
    latentsFigFilenamePattern = \
        "figures/{:08d}_simulation_latents.{{:s}}".format(simResNumber)
    cifFigFilenamePattern = \
        "figures/{:08d}_simulation_cif_trial{:03d}_neuron{:03d}.{{:s}}".format(simResNumber,
                                                                               exampleTrial,
                                                                              exampleNeuron)
    spikesTimesFigFilenamePattern = \
        "figures/{:08d}_simulation_spikesTimes.{{:s}}".format(simResNumber)
    spikesRatesFigFilenamePattern = \
        "figures/{:08d}_simulation_spikesRates.{{:s}}".format(simResNumber)
    ksTestTimeRescalingFigFilenamePattern = \
        "figures/{:08d}_simulation_ksTestTimeRescaling.{{:s}}".format(simResNumber)
    rocFigFilenamePattern = \
        "figures/{:08d}_simulation_rocAnalysis_trial{:03d}_neuron{:03d}.{{:s}}".format(simResNumber, exampleTrial, exampleNeuron)

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    times = simRes["times"]
    latents = simRes["latents"]
    latentsMeans = simRes["latentsMeans"]
    latentsSTDs = simRes["latentsSTDs"]
    cifValues = simRes["cifValues"]
    spikes = simRes["spikes"]

    fig = plot.svGPFA.plotUtilsPlotly.getSimulatedLatentsPlotPlotly(trialsTimes=times, latentsSamples=latents, latentsMeans=latentsMeans, latentsSTDs=latentsSTDs)
    fig.write_image(latentsFigFilenamePattern.format("png"))
    fig.write_html(latentsFigFilenamePattern.format("html"))
    fig.show()

    timesCIFToPlot = times[exampleTrial]
    valuesCIFToPlot = cifValues[exampleTrial][exampleNeuron]
    title = "Trial {:d}, Neuron {:d}".format(exampleTrial, exampleNeuron)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotCIFPlotly(times=timesCIFToPlot, values=valuesCIFToPlot, title=title)
    fig.write_image(cifFigFilenamePattern.format("png"))
    fig.write_html(cifFigFilenamePattern.format("html"))
    fig.show()

    fig = plot.svGPFA.plotUtilsPlotly.getSimulatedSpikesTimesPlotPlotly(spikesTimes=spikes)
    fig.write_image(spikesTimesFigFilenamePattern.format("png"))
    fig.write_html(spikesTimesFigFilenamePattern.format("html"))
    fig.show()

    spikesRates = utils.svGPFA.miscUtils.computeSpikeRates(trialsTimes=times, spikesTimes=spikes)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotSpikeRatesForAllTrialsAndAllNeurons(spikesRates=spikesRates)
    fig.write_image(spikesRatesFigFilenamePattern.format("png"))
    fig.write_html(spikesRatesFigFilenamePattern.format("html"))
    fig.show()

    oneTrialCIFTimes = torch.arange(0, T, dtCIF)
    cifTimes = torch.unsqueeze(torch.ger(torch.ones(nTrials), oneTrialCIFTimes), dim=2)
    cifTimesKS = cifTimes[exampleTrial,:,0]
    cifValuesKS = cifValues[exampleTrial][exampleNeuron]
    spikesTimesKS = spikes[exampleTrial][exampleNeuron]
    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = stats.pointProcess.tests.KSTestTimeRescalingNumericalCorrection(spikesTimes=spikesTimesKS, cifTimes=cifTimesKS, cifValues=cifValuesKS, gamma=nResamplesKSTest)
    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(exampleTrial, exampleNeuron, len(spikesTimesKS))
    fig = plot.svGPFA.plotUtils.plotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx, estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb, title=title)
    plt.savefig(fname=ksTestTimeRescalingFigFilenamePattern.format("png"))
    plt.close("all")
    # p = ggplotly(p)
    # plotly.offline.plot(pSpikes, filename=ksTestTimeRescalingFigFilenamePattern%"html")

    pk = cifValuesKS*dtCIF
    bins = pd.interval_range(start=0, end=T, periods=len(pk))
    # start binning spikes using pandas
    cutRes, _ = pd.cut(spikesTimesKS, bins=bins, retbins=True)
    Y = torch.from_numpy(cutRes.value_counts().values)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, pk, pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    title = "Trial {:d}, Neuron {:d}".format(exampleTrial, exampleNeuron)
    fig = plot.svGPFA.plotUtils.plotResROCAnalysis(fpr=fpr, tpr=tpr, auc=roc_auc, title=title)
    plt.savefig(fname=rocFigFilenamePattern.format("png"))
    # p = ggplotly(p)
    # plotly.offline.plot(pSpikes, filename=rocFigFilenamePattern%"html")

    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
