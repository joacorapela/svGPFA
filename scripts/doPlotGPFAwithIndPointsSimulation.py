
import pdb
import sys
import os
import random
import torch
import plotly
import plotly.io as pio
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
import utils.svGPFA.configUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simResNumber", help="Simulation result number", type=int)
    parser.add_argument("--latentToPlot", help="Latent to plot", type=int, default=0)
    parser.add_argument("--trialToPlot", help="Trial to plot", type=int, default=0)
    parser.add_argument("--neuronToPlot", help="Neuron to plot", type=int, default=0)
    parser.add_argument("--nResamplesKSTest", help="Number of resamples for KS test", type=int, default=10)
    args = parser.parse_args()

    simResNumber = args.simResNumber
    latentToPlot = args.latentToPlot
    trialToPlot = args.trialToPlot
    neuronToPlot = args.neuronToPlot
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
    CFilename = simInitConfig["embedding_params"]["C_filename"]
    dFilename = simInitConfig["embedding_params"]["d_filename"]
    C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(CFilename=CFilename, dFilename=dFilename)

    simResFilename = "results/{:08d}_simRes.pickle".format(simResNumber)
    latentFigFilenamePattern = \
        "figures/{:08d}_simulation_latent_trial{:03d}_latent{:03d}.{{:s}}".format(simResNumber, trialToPlot, latentToPlot)
    embeddingFigFilenamePattern = \
        "figures/{:08d}_simulation_embedding_trial{:03d}_neuron{:03d}.{{:s}}".format(simResNumber, trialToPlot, neuronToPlot)
    cifFigFilenamePattern = \
        "figures/{:08d}_simulation_cif_trial{:03d}_neuron{:03d}.{{:s}}".format(simResNumber, trialToPlot, neuronToPlot)
    spikesTimesFigFilenamePattern = \
        "figures/{:08d}_simulation_spikesTimes_trial{:03d}.{{:s}}".format(simResNumber, trialToPlot)
    spikesRatesFigFilenamePattern = \
       "figures/{:08d}_simulation_spikesRates.{{:s}}".format(simResNumber)
    ksTestTimeRescalingFigFilenamePattern = \
        "figures/{:08d}_simulation_ksTestTimeRescaling_trial{:03d}_neuron{:03d}.{{:s}}".format(simResNumber, trialToPlot, neuronToPlot)
    rocFigFilenamePattern = \
        "figures/{:08d}_simulation_rocAnalysis_trial{:03d}_neuron{:03d}.{{:s}}".format(simResNumber, trialToPlot, neuronToPlot)

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    times = simRes["latentsTrialsTimes"]
    latentsSamples = simRes["latentsSamples"]
    latentsMeans = simRes["latentsMeans"]
    latentsSTDs = simRes["latentsSTDs"]
    cifValues = simRes["cifValues"]
    spikes = simRes["spikes"]

    pio.renderers.default = "browser"

    timesLatentToPlot = times[trialToPlot]
    latentSamplesToPlot = latentsSamples[trialToPlot][latentToPlot,:]
    latentMeansToPlot = latentsMeans[trialToPlot][latentToPlot,:]
    latentSTDsToPlot = latentsSTDs[trialToPlot][latentToPlot,:]
    title = "Trial {:d}, Latent {:d}".format(trialToPlot, latentToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getSimulatedLatentPlot(times=timesLatentToPlot, latentSamples=latentSamplesToPlot, latentMeans=latentMeansToPlot, latentSTDs=latentSTDsToPlot, title=title)
    fig.write_image(latentFigFilenamePattern.format("png"))
    fig.write_html(latentFigFilenamePattern.format("html"))
    fig.show()

    # embeddingSamples[r], embeddingMeans[r], embeddingSTDs \in nNeurons x nSamples
    embeddingSamples = [torch.matmul(C, latentsSamples[r])+d for r in range(nTrials)]
    embeddingMeans = [torch.matmul(C, latentsMeans[r])+d for r in range(nTrials)]
    embeddingSTDs = [torch.matmul(C, latentsSTDs[r]) for r in range(nTrials)]
    timesEmbeddingToPlot = times[trialToPlot]
    embeddingSamplesToPlot = embeddingSamples[trialToPlot][neuronToPlot,:]
    embeddingMeansToPlot = embeddingMeans[trialToPlot][neuronToPlot,:]
    embeddingSTDsToPlot = embeddingSTDs[trialToPlot][neuronToPlot,:]
    title = "Trial {:d}, Neuron {:d}".format(trialToPlot, neuronToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getSimulatedEmbeddingPlot(times=timesEmbeddingToPlot, samples=embeddingSamplesToPlot, means=embeddingMeansToPlot, stds=embeddingSTDsToPlot, title=title)
    fig.write_image(embeddingFigFilenamePattern.format("png"))
    fig.write_html(embeddingFigFilenamePattern.format("html"))
    fig.show()

    timesCIFToPlot = times[trialToPlot]
    valuesCIFToPlot = cifValues[trialToPlot][neuronToPlot]
    title = "Trial {:d}, Neuron {:d}".format(trialToPlot, neuronToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotCIF(times=timesCIFToPlot, values=valuesCIFToPlot, title=title)
    fig.write_image(cifFigFilenamePattern.format("png"))
    fig.write_html(cifFigFilenamePattern.format("html"))
    fig.show()

#     spikesToPlot = spikes[trialToPlot]
#     title = "Trial {:d}".format(trialToPlot)
#     fig = plot.svGPFA.plotUtilsPlotly.getSpikesTimesPlotOneTrial(spikes_times=spikesToPlot, title=title)
#     fig.write_image(spikesTimesFigFilenamePattern.format("png"))
#     fig.write_html(spikesTimesFigFilenamePattern.format("html"))
#     fig.show()

    spikesRates = utils.svGPFA.miscUtils.computeSpikeRates(trialsTimes=times, spikesTimes=spikes)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotSpikeRatesForAllTrialsAndAllNeurons(spikesRates=spikesRates)
    fig.write_image(spikesRatesFigFilenamePattern.format("png"))
    fig.write_html(spikesRatesFigFilenamePattern.format("html"))
    fig.show()

    oneTrialCIFTimes = torch.arange(0, T, dtCIF)
    cifTimes = torch.unsqueeze(torch.ger(torch.ones(nTrials), oneTrialCIFTimes), dim=2)
    cifTimesKS = cifTimes[trialToPlot,:,0]
    cifValuesKS = cifValues[trialToPlot][neuronToPlot]
    spikesTimesKS = spikes[trialToPlot][neuronToPlot]
    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = stats.pointProcess.tests.KSTestTimeRescalingNumericalCorrection(spikesTimes=spikesTimesKS, cifTimes=cifTimesKS, cifValues=cifValuesKS, gamma=nResamplesKSTest)
    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(trialToPlot, neuronToPlot, len(spikesTimesKS))
    fig = plot.svGPFA.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(
        diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx,
        estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb,
        title=title)
    fig.write_image(ksTestTimeRescalingFigFilenamePattern.format("png"))
    fig.write_html(ksTestTimeRescalingFigFilenamePattern.format("html"))
    fig.show()

    pk = cifValuesKS*dtCI
    bins = pd.interval_range(start=0, end=T, periods=len(pk))
    cutRes, _ = pd.cut(spikesTimesKS, bins=bins, retbins=True)
    Y = torch.from_numpy(cutRes.value_counts().values)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, pk, pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    title = "Trial {:d}, Neuron {:d}".format(trialToPlot, neuronToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotResROCAnalysis(fpr=fpr, tpr=tpr, auc=roc_auc, title=title)
    fig.write_image(rocFigFilenamePattern.format("png"))
    fig.write_html(rocFigFilenamePattern.format("html"))
    fig.show()


    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
