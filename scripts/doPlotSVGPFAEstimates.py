
import sys
import os
import pdb
import math
from scipy.io import loadmat
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
sys.path.append("../src")
import stats.pointProcess.tests
import utils.svGPFA.configUtils
import utils.svGPFA.initUtils
import utils.svGPFA.miscUtils
import plot.svGPFA.plotUtils
import plot.svGPFA.plotUtilsPlotly

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--trialToPlot", help="trial to plot", type=int, default=0)
    parser.add_argument("--neuronToPlot", help="neuron to plot", type=int, default=0)
    parser.add_argument("--dtCIF", help="neuron to plot", type=float, default=1e-3)
    parser.add_argument("--ksTestGamma", help="gamma value for KS test", type=int, default=10)
    parser.add_argument("--nTestPoints", help="number of test points where to plot latents", type=int, default=2000)
    args = parser.parse_args()

    estResNumber = args.estResNumber
    trialToPlot = args.trialToPlot
    neuronToPlot = args.neuronToPlot
    dtCIF = args.dtCIF
    ksTestGamma = args.ksTestGamma
    nTestPoints = args.nTestPoints

    estimResMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(estResNumber)
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)
    lowerBoundHistVsIterNoFigFilenamePattern = "figures/{:08d}_lowerBoundHistVsIterNo.{{:s}}".format(estResNumber)
    lowerBoundHistVsElapsedTimeFigFilenamePattern = "figures/{:08d}_lowerBoundHistVsElapsedTime.{{:s}}".format(estResNumber)
    latentsFigFilenamePattern = "figures/{:08d}_estimatedLatent_trial{:03d}_neuron{:03d}.{{:s}}".format(estResNumber, trialToPlot, neuronToPlot)
    ksTestTimeRescalingNumericalCorrectionFigFilename = "figures/{:08d}_ksTestTimeRescaling_numericalCorrection_trial{:03d}_neuron{:03d}.png".format(estResNumber, trialToPlot, neuronToPlot)
    trueAndEstimatedCIFsFigFilenamePattern = "figures/{:08d}_trueAndEstimatedCIFs_trial{:03d}_neuron{:03d}.{{:s}}".format(estResNumber, trialToPlot, neuronToPlot)
    rocFigFilename = "figures/{:08d}_rocAnalysis_trial{:03d}_neuron{:03d}.png".format(estResNumber, trialToPlot, neuronToPlot)
    kernelsParamsFigFilenamePattern = "figures/{:08d}_trueAndEstimatedKernelsParams.{{:s}}".format(estResNumber)
    embeddingParamsFigFilenamePattern = "figures/{:08d}_trueAndEstimatedEmbeddingParams.{{:s}}".format(estResNumber)
    ksTestTimeRescalingAnalyticalCorrectionFigFilename = "figures/{:08d}_ksTestTimeRescaling_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(estResNumber, trialToPlot, neuronToPlot)
    timeRescalingDiffCDFsFigFilename = "figures/{:08d}_timeRescalingDiffCDFs_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(estResNumber, trialToPlot, neuronToPlot)
    timeRescaling1LagScatterPlotFigFilename = "figures/{:08d}_timeRescaling1LagScatterPlot_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(estResNumber, trialToPlot, neuronToPlot)
    timeRescalingACFFigFilename = "figures/{:08d}_timeRescalingACF_analyticalCorrection_trial{:03d}_neuron{:03d}.png".format(estResNumber, trialToPlot, neuronToPlot)

    estimResConfig = configparser.ConfigParser()
    estimResConfig.read(estimResMetaDataFilename)
    simResNumber = int(estimResConfig["simulation_params"]["simResNumber"])
    simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]
    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nLatents = int(simInitConfig["control_variables"]["nLatents"])
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    dtCIF = float(simInitConfig["control_variables"]["dtCIF"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    tKernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=True)
    kernelsTypes = [type(tKernels[k]).__name__ for k in range(nLatents)]
    tKernelsParams = utils.svGPFA.initUtils.getKernelsParams0(kernels=tKernels, noiseSTD=0.0)
    simResFilename = simResConfig["simulation_results"]["simResFilename"]
    CFilename = simInitConfig["embedding_params"]["C_filename"]
    dFilename = simInitConfig["embedding_params"]["d_filename"]
    C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(CFilename=CFilename, dFilename=dFilename)
    tIndPointsLocs = utils.svGPFA.configUtils.getIndPointsLocs0(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    trueLatentsSamples = simRes["latentsSamples"]
    simCIFsValues = simRes["cifValues"]
    trueLatentsSamples = [trueLatentsSamples[r][:nLatents,:] for r in range(nTrials)]
    trueLatentsTimes = simRes["latentsTrialsTimes"]
    trueLatentsMeans = simRes["latentsMeans"]
    trueLatentsMeans = [trueLatentsMeans[r][:nLatents,:] for r in range(nTrials)]
    trueLatentsSTDs = simRes["latentsSTDs"]
    trueLatentsSTDs = [trueLatentsSTDs[r][:nLatents,:] for r in range(nTrials)]
    timesTrueValues = torch.linspace(0, torch.max(torch.tensor(trialsLengths)), trueLatentsSamples[0].shape[1])
    testTimes = torch.linspace(0, torch.max(torch.tensor(spikesTimes[0][0])), nTestPoints)

    with open(modelSaveFilename, "rb") as f: estResults = pickle.load(f)
    lowerBoundHist = estResults["lowerBoundHist"]
    elapsedTimeHist = estResults["elapsedTimeHist"]
    model = estResults["model"]

    # plot lower bound history
    fig = plot.svGPFA.plotUtilsPlotly.getPlotLowerBoundHist(lowerBoundHist=lowerBoundHist)
    fig.write_image(lowerBoundHistVsIterNoFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundHistVsIterNoFigFilenamePattern.format("html"))

    fig = plot.svGPFA.plotUtilsPlotly.getPlotLowerBoundHist(elapsedTimeHist=elapsedTimeHist, lowerBoundHist=lowerBoundHist)
    fig.write_image(lowerBoundHistVsElapsedTimeFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundHistVsElapsedTimeFigFilenamePattern.format("html"))

    # plot true and estimated latents
    testMuK, testVarK = model.predictLatents(times=trueLatentsTimes[0])
    eIndPointsLocs = model.getIndPointsLocs()
    fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedLatents(tTimes=trueLatentsTimes[0], tLatentsSamples=trueLatentsSamples, tLatentsMeans=trueLatentsMeans, tLatentsSTDs=trueLatentsSTDs, tIndPointsLocs=tIndPointsLocs, eTimes=trueLatentsTimes[0], eLatentsMeans=testMuK, eLatentsSTDs=torch.sqrt(testVarK), eIndPointsLocs=eIndPointsLocs, trialToPlot=trialToPlot)
    fig.write_image(latentsFigFilenamePattern.format("png"))
    fig.write_html(latentsFigFilenamePattern.format("html"))

    # KS test time rescaling with numerical correction
    T = torch.tensor(trialsLengths).max()
    oneTrialCIFTimes = torch.arange(0, T, dtCIF)
    cifTimes = torch.unsqueeze(torch.ger(torch.ones(nTrials), oneTrialCIFTimes), dim=2)
    with torch.no_grad():
#         emcifValues = model.computeCIFsMeans(times=cifTimes)
#         epmcifValues = model.computeExpectedCIFs(times=cifTimes)
        epcifValues = model.computeExpectedPosteriorCIFs(times=oneTrialCIFTimes)
    spikesTimesKS = spikesTimes[trialToPlot][neuronToPlot]
    cifTimesKS = cifTimes[trialToPlot,:,0]
    cifValuesKS = epcifValues[trialToPlot][neuronToPlot]

    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(trialToPlot, neuronToPlot, len(spikesTimesKS))

    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = stats.pointProcess.tests.KSTestTimeRescalingNumericalCorrection(spikesTimes=spikesTimesKS, cifTimes=oneTrialCIFTimes, cifValues=cifValuesKS, gamma=ksTestGamma)
    plot.svGPFA.plotUtils.plotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx, estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb, figFilename=ksTestTimeRescalingNumericalCorrectionFigFilename, title=title)
    plt.close("all")

    # CIF
    fig = plot.svGPFA.plotUtilsPlotly.getPlotSimulatedAndEstimatedCIFs(
        tTimes=timesTrueValues, 
        tCIF=simCIFsValues[trialToPlot][neuronToPlot], 
        tLabel="True", 
        eMeanTimes=oneTrialCIFTimes, 
#         eMeanCIF=emcifValues[trialToPlot][neuronToPlot], 
        eMeanCIF=epcifValues[trialToPlot][neuronToPlot], 
        eMeanLabel="Mean", 
#         ePosteriorMeanTimes=oneTrialCIFTimes, 
#         ePosteriorMeanCIF=epmcifValues[trialToPlot][neuronToPlot], 
#         ePosteriorMeanLabel="Posterior Mean",
        title=title)
    fig.write_image(trueAndEstimatedCIFsFigFilenamePattern.format("png"))
    fig.write_html(trueAndEstimatedCIFsFigFilenamePattern.format("html"))

    # ROC predictive analysis
    cifValuesKS_upsampled = torch.from_numpy(np.interp(x=np.arange(0, T, 1e-3),
                                                       xp=np.arange(0, T,
                                                                    dtCIF),
                                                       fp=cifValuesKS))
    pk = cifValuesKS_upsampled*dtCIF
    bins = pd.interval_range(start=0, end=int(T), periods=len(pk))
    cutRes, _ = pd.cut(spikesTimesKS, bins=bins, retbins=True)
    Y = torch.from_numpy(cutRes.value_counts().values)
    Y[Y>1] = 1
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, pk, pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plot.svGPFA.plotUtils.plotResROCAnalysis(fpr=fpr, tpr=tpr, auc=roc_auc, title=title, figFilename=rocFigFilename)
    plt.close("all")

    # plot model params
    mKernelsParams = model.getKernelsParams()
    fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedKernelsParams(
        kernelsTypes=kernelsTypes,
        trueKernelsParams=tKernelsParams,
        estimatedKernelsParams=mKernelsParams)
    fig.write_image(kernelsParamsFigFilenamePattern.format("png"))
    fig.write_html(kernelsParamsFigFilenamePattern.format("html"))

    # tLatentsMeansFuncs = utils.svGPFA.configUtils.getLatentsMeansFuncs(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
    # trialsTimes = utils.svGPFA.miscUtils.getTrialsTimes(trialsLengths=trialsLengths, dt=dtCIF)
    # tLatentsMeans = utils.svGPFA.miscUtils.getLatentsMeanFuncsSamples(latentsMeansFuncs=tLatentsMeansFuncs, trialsTimes=trialsTimes, dtype=C.dtype)
    # kernelsParams = model.getKernelsParams()
    # kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=True)
    # with torch.no_grad():
    #     latentsMeans, _ = model.predictLatents(newTimes=trialsTimes[0])
    # fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedKernelsParams(trueKernels=kernels, estimatedKernelsParams=kernelsParams)
    # fig.write_image(kernelsParamsFigFilenamePattern.format("png"))
    # fig.write_html(kernelsParamsFigFilenamePattern.format("html"))

    estimatedC, estimatedD = model.getSVEmbeddingParams()
    fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedEmbeddingParams(trueC=C.numpy(), trueD=d.numpy(), estimatedC=estimatedC.numpy(), estimatedD=estimatedD.numpy())
    fig.write_image(embeddingParamsFigFilenamePattern.format("png"))
    fig.write_html(embeddingParamsFigFilenamePattern.format("html"))

    # KS test time rescaling with analytical correction
#     t0 = math.floor(cifTimesKS.min())
#     tf = math.ceil(cifTimesKS.max())
#     dt = (cifTimesKS[1]-cifTimesKS[0]).item()
#     utSRISIs, uCDF, cb, utRISIs = stats.pointProcess.tests.KSTestTimeRescalingAnalyticalCorrectionUnbinned(spikesTimes=spikesTimesKS, cifValues=cifValuesKS, t0=t0, tf=tf, dt=dt)
#     sUTRISIs, _ = torch.sort(utSRISIs)

#     plot.svGPFA.plotUtils.plotResKSTestTimeRescalingAnalyticalCorrection(sUTRISIs=sUTRISIs, uCDF=uCDF, cb=cb, title=title, figFilename=ksTestTimeRescalingAnalyticalCorrectionFigFilename)
#     plt.close("all")
#     plot.svGPFA.plotUtils.plotDifferenceCDFs(sUTRISIs=sUTRISIs, uCDF=uCDF, cb=cb, figFilename=timeRescalingDiffCDFsFigFilename),
#     plt.close("all")
#     plot.svGPFA.plotUtils.plotScatter1Lag(x=utRISIs, title=title, figFilename=timeRescaling1LagScatterPlotFigFilename)
#     plt.close("all")
# #     acfRes, confint = statsmodels.tsa.stattools.acf(x=utRISIs, unbiased=True, alpha=0.05)
# #     plot.svGPFA.plotUtils.plotACF(acf=acfRes, Fs=1/dt, confint=confint, title=title, figFilename=timeRescalingACFFigFilename),
# #     plt.close("all")

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
