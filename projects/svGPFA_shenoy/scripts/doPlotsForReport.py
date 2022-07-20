
import sys
import pdb
import math
import numpy as np
import torch
import pickle
import argparse
import configparser
import pandas as pd
import sklearn.metrics
# import statsmodels.tsa.stattools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append("../../src")
import stats.pointProcess.tests
import utils.svGPFA.configUtils
import utils.svGPFA.initUtils
import utils.svGPFA.miscUtils
import plot.svGPFA.plotUtils
import plot.svGPFA.plotUtilsPlotly
import shenoyUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--latentToPlot", help="trial to plot", type=int, default=0)
    parser.add_argument("--neuronToPlot", help="neuron to plot", type=int, default=0)
    parser.add_argument("--trialToPlot", help="trial to plot", type=int, default=0)
    parser.add_argument("--dtCIF", help="neuron to plot", type=float, default=1.0)
    parser.add_argument("--ksTestGamma", help="gamma value for KS test", type=int, default=10)
    parser.add_argument("--nTestPoints", help="number of test points where to plot latents", type=int, default=2000)
    parser.add_argument("--location", help="location to analyze", type=int,
                        default=0)
    parser.add_argument("--trials_indices", help="trials indices to analyze",
                        default="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]")
    parser.add_argument("--from_time", help="starting spike analysis time",
                        type=float, default=750.0)
    parser.add_argument("--to_time", help="ending spike analysis time",
                        type=float, default=2500.0)
    parser.add_argument("--min_nSpikes_perNeuron_perTrial",
                        help="min number of spikes per neuron per trial",
                        type=int, default=1)
    parser.add_argument("--data_filename", help="data filename",
                        default="~/dev/research/gatsby-swc/datasets/george20040123_hnlds.mat")
    args = parser.parse_args()

    estResNumber = args.estResNumber
    latentToPlot = args.latentToPlot
    neuronToPlot = args.neuronToPlot
    trialToPlot = args.trialToPlot
    dtCIF = args.dtCIF
    ksTestGamma = args.ksTestGamma
    nTestPoints = args.nTestPoints
    location = args.location
    trials_indices = [int(str) for str in args.trials_indices[1:-1].split(",")]
    from_time = args.from_time
    to_time = args.to_time
    min_nSpikes_perNeuron_perTrial = args.min_nSpikes_perNeuron_perTrial
    data_filename = args.data_filename

    estimResMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(estResNumber)
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)
    lowerBoundHistVsIterNoFigFilenamePattern = "figures/{:08d}_lowerBoundHistVSIterNo.{{:s}}".format(estResNumber)
    lowerBoundHistVsElapsedTimeFigFilenamePattern = "figures/{:08d}_lowerBoundHistVsElapsedTime.{{:s}}".format(estResNumber)
    latentsFigFilenamePattern = "figures/{:08d}_estimatedLatent_latent{:03d}.{{:s}}".format(estResNumber, latentToPlot)
    embeddingsFigFilenamePattern = "figures/{:08d}_estimatedEmbedding_neuron{:d}.{{:s}}".format(estResNumber, neuronToPlot)
    embeddingParamsFigFilenamePattern = "figures/{:08d}_estimatedEmbeddingParams.{{:s}}".format(estResNumber)
    CIFFigFilenamePattern = "figures/{:08d}_CIF_neuron{:03d}.{{:s}}".format(estResNumber, neuronToPlot)
    ksTestTimeRescalingNumericalCorrectionFigFilename = "figures/{:08d}_ksTestTimeRescaling_numericalCorrection_trial{:03d}_neuron{:03d}.png".format(estResNumber, trialToPlot, neuronToPlot)
    rocFigFilename = "figures/{:08d}_rocAnalysis_trial{:03d}_neuron{:03d}.png".format(estResNumber, trialToPlot, neuronToPlot)
    kernelsParamsFigFilenamePattern = "figures/{:08d}_estimatedKernelsParams.{{:s}}".format(estResNumber)

    spikes_times, neurons_indices = \
            shenoyUtils.getSpikesTimes(data_filename=data_filename,
                                       trials_indices=trials_indices,
                                       location=location,
                                       from_time=from_time,
                                       to_time=to_time,
                                       min_nSpikes_perNeuron_perTrial=
                                        min_nSpikes_perNeuron_perTrial)

    estimResConfig = configparser.ConfigParser()
    estimResConfig.read(estimResMetaDataFilename)
    nLatents = int(estimResConfig["data_params"]["nLatents"])
    from_time = float(estimResConfig["data_params"]["from_time"])
    to_time = float(estimResConfig["data_params"]["to_time"])
    trials = [float(str) for str in
              estimResConfig["data_params"]["trials_indices"][1:-1].split(",")]
    nTrials = len(trials)
    trial_times = torch.arange(from_time, to_time, dtCIF)
    trial_times_numpy = trial_times.detach().numpy()

    with open(modelSaveFilename, "rb") as f: estResults = pickle.load(f)
    lowerBoundHist = estResults["lowerBoundHist"]
    elapsedTimeHist = estResults["elapsedTimeHist"]
    model = estResults["model"]
    neurons_indices = estResults["neurons_indices"]
    neuronToPlot_index = torch.nonzero(torch.tensor(neurons_indices)==neuronToPlot)
    neurons_indices_str = "".join(str(i)+" " for i in neurons_indices)
    if len(neuronToPlot_index)==0:
        raise ValueError("Neuron {:d} is not valid. Valid neurons are ".format(neuronToPlot) + neurons_indices_str)

    # plot lower bound history
    fig = plot.svGPFA.plotUtilsPlotly.getPlotLowerBoundHist(lowerBoundHist=lowerBoundHist)
    fig.write_image(lowerBoundHistVsIterNoFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundHistVsIterNoFigFilenamePattern.format("html"))

    fig = plot.svGPFA.plotUtilsPlotly.getPlotLowerBoundHist(elapsedTimeHist=elapsedTimeHist, lowerBoundHist=lowerBoundHist)
    fig.write_image(lowerBoundHistVsElapsedTimeFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundHistVsElapsedTimeFigFilenamePattern.format("html"))

    # plot estimated latent across trials
    testMuK, testVarK = model.predictLatents(times=trial_times)
    indPointsLocs = model.getIndPointsLocs()
    fig = plot.svGPFA.plotUtilsPlotly.getPlotLatentAcrossTrials(times=trial_times, latentsMeans=testMuK, latentsSTDs=torch.sqrt(testVarK), indPointsLocs=indPointsLocs, latentToPlot=latentToPlot, xlabel="Time (msec)")
    fig.write_image(latentsFigFilenamePattern.format("png"))
    fig.write_html(latentsFigFilenamePattern.format("html"))

    # plot embedding
    embeddingMeans, embeddingVars = model.predictEmbedding(times=trial_times)
    embeddingMeans = embeddingMeans.detach().numpy()
    embeddingVars = embeddingVars.detach().numpy()
    title = "Neuron {:d}".format(neuronToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(times=trial_times_numpy, embeddingsMeans=embeddingMeans[:,:,neuronToPlot], embeddingsSTDs=np.sqrt(embeddingVars[:,:,neuronToPlot]), title=title)
    fig.write_image(embeddingsFigFilenamePattern.format("png"))
    fig.write_html(embeddingsFigFilenamePattern.format("html"))

    # calculate expected CIF values (for KS test and CIF plots)
    with torch.no_grad():
        ePosCIFValues = model.computeExpectedPosteriorCIFs(times=trial_times)
    spikesTimesKS = spikes_times[trialToPlot][neuronToPlot]
    cifValuesKS = ePosCIFValues[trialToPlot][neuronToPlot]
    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(trialToPlot, neuronToPlot, len(spikesTimesKS))

    # CIF
    fig = plot.svGPFA.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(times=trials_times, cif_values=ePosCIFValues, neuron_index=neuronToPlot)
    fig.write_image(CIFFigFilenamePattern.format("png"))
    fig.write_html(CIFFigFilenamePattern.format("html"))

    # plot KS test time rescaling (numerical correction)
    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = stats.pointProcess.tests.KSTestTimeRescalingNumericalCorrection(spikesTimes=spikesTimesKS, cifTimes=trial_times, cifValues=cifValuesKS, gamma=ksTestGamma)
    plot.svGPFA.plotUtils.plotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx, estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb, figFilename=ksTestTimeRescalingNumericalCorrectionFigFilename, title=title)
    plt.close("all")

    # ROC predictive analysis
    pk = cifValuesKS*dtCIF
    bins = pd.interval_range(start=int(min(trial_times)),
                             end=int(max(trial_times)), periods=len(pk))
    cutRes, _ = pd.cut(spikesTimesKS, bins=bins, retbins=True)
    Y = torch.from_numpy(cutRes.value_counts().values)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, pk, pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plot.svGPFA.plotUtils.plotResROCAnalysis(fpr=fpr, tpr=tpr, auc=roc_auc, title=title, figFilename=rocFigFilename)
    plt.close("all")

    # plot embedding parameters
    estimatedC, estimatedD = model.getSVEmbeddingParams()
    fig = plot.svGPFA.plotUtilsPlotly.getPlotEmbeddingParams(C=estimatedC.numpy(), d=estimatedD.numpy())
    fig.write_image(embeddingParamsFigFilenamePattern.format("png"))
    fig.write_html(embeddingParamsFigFilenamePattern.format("html"))

    # plot kernel parameters
    kernelsParams = model.getKernelsParams()
    kernelsTypes = [type(kernel).__name__ for kernel in model.getKernels()]
    fig = plot.svGPFA.plotUtilsPlotly.getPlotKernelsParams(
        kernelsTypes=kernelsTypes, kernelsParams=kernelsParams)
    fig.write_image(kernelsParamsFigFilenamePattern.format("png"))
    fig.write_html(kernelsParamsFigFilenamePattern.format("html"))

#     pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
