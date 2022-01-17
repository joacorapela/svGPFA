
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
import scipy.io
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

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--latentToPlot", help="trial to plot", type=int, default=0)
    parser.add_argument("--neuronToPlot", help="neuron to plot", type=int, default=0)
    parser.add_argument("--dtCIF", help="neuron to plot", type=float, default=1.0)
    parser.add_argument("--ksTestGamma", help="gamma value for KS test", type=int, default=10)
    parser.add_argument("--nTestPoints", help="number of test points where to plot latents", type=int, default=2000)
    parser.add_argument("--modelFilenamePattern", help="model filename pattern",
                        default="../../../matlabCode/working/scripts/results/{:08d}-pointProcessEstimationRes.mat")
    args = parser.parse_args()

    estResNumber = args.estResNumber
    latentToPlot = args.latentToPlot
    neuronToPlot = args.neuronToPlot
    dtCIF = args.dtCIF
    ksTestGamma = args.ksTestGamma
    nTestPoints = args.nTestPoints
    modelFilenamePattern = args.modelFilenamePattern

    lowerBoundHistVsIterNoFigFilenamePattern = "../../../matlabCode/working/scripts/figures/{:08d}_lowerBoundHistVSIterNo.{{:s}}".format(estResNumber)
    lowerBoundHistVsElapsedTimeFigFilenamePattern = "../../../matlabCode/working/scripts/figures/{:08d}_lowerBoundHistVsElapsedTime.{{:s}}".format(estResNumber)
    latentsFigFilenamePattern = "../../../matlabCode/working/scripts/figures/{:08d}_estimatedLatent_latent{:03d}.{{:s}}".format(estResNumber, latentToPlot)
    embeddingsFigFilenamePattern = "../../../matlabCode/working/scripts/figures/{:08d}_estimatedEmbedding_neuron{:d}.{{:s}}".format(estResNumber, neuronToPlot)
    embeddingParamsFigFilenamePattern = "../../../matlabCode/working/scripts/figures/{:08d}_estimatedEmbeddingParams.{{:s}}".format(estResNumber)
    kernelsParamsFigFilenamePattern = "../../../matlabCode/working/scripts/figures/{:08d}_estimatedKernelsParams.{{:s}}".format(estResNumber)

    # estimResConfig = configparser.ConfigParser()
    # estimResConfig.read(estimResMetaDataFilename)
    # nLatents = int(estimResConfig["data_params"]["nLatents"])
    # from_time = float(estimResConfig["data_params"]["from_time"])
    # to_time = float(estimResConfig["data_params"]["to_time"])
    # trials = [float(str) for str in
    #           estimResConfig["data_params"]["trials_indices"][1:-1].split(",")]
    # nTrials = len(trials)
    # trial_times = torch.arange(from_time, to_time, dtCIF)

    modelFilename = modelFilenamePattern.format(estResNumber)
    res = scipy.io.loadmat(modelFilename)

    trial_times = res["latentsTimes"].squeeze().tolist()

    # lowerBound
    lowerBoundHist = res["lowerBound"].squeeze().tolist()
    elapsedTimeHist = res['m']['iterationTime'][0,0].squeeze().tolist()

    # latents
    times = res["latentsTimes"].squeeze().tolist()
    latentsMeans = res["meanEstimatedLatents"].transpose((2, 0, 1))
    latentsVars = res["varEstimatedLatents"].transpose((2, 0, 1))

    # embedding
    C = res["m"][0,0]["prs"][0,0]["C"]
    d = res["m"][0,0]["prs"][0,0]["b"]
    embeddingMeans = np.matmul(latentsMeans, C.T) + np.reshape(d, (1, 1, len(d)))
    embeddingVars = np.matmul(latentsVars, (C.T)**2)

    # kernels params
    nLatents = C.shape[1]
    kernelsTypes = ["ExponentialQuadraticKernel" for k in range(nLatents)]
    kernelsParams = [[] for k in range(nLatents)]
    for k in range(nLatents):
        kernelsParams[k] = [res["m"][0,0]["kerns"][0,k]["hprs"][0,0].squeeze()]

    # with open(modelSaveFilename, "rb") as f: estResults = pickle.load(f)
    # lowerBoundHist = estResults["lowerBoundHist"]
    # model = estResults["model"]
    # neurons_indices = estResults["neurons_indices"]
    # neuronToPlot_index = torch.nonzero(torch.tensor(neurons_indices)==neuronToPlot)
    # neurons_indices_str = "".join(str(i)+" " for i in neurons_indices)
    # if len(neuronToPlot_index)==0:
    #     raise ValueError("Neuron {:d} is not valid. Valid neurons are ".format(neuronToPlot) + neurons_indices_str)

    # plot lower bound history
    fig = plot.svGPFA.plotUtilsPlotly.getPlotLowerBoundHist(lowerBoundHist=lowerBoundHist)
    fig.write_image(lowerBoundHistVsIterNoFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundHistVsIterNoFigFilenamePattern.format("html"))

    fig = plot.svGPFA.plotUtilsPlotly.getPlotLowerBoundHist(elapsedTimeHist=elapsedTimeHist, lowerBoundHist=lowerBoundHist)
    fig.write_image(lowerBoundHistVsElapsedTimeFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundHistVsElapsedTimeFigFilenamePattern.format("html"))

    title = "Neuron {:d}".format(neuronToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(times=trial_times, embeddingsMeans=embeddingMeans[:,:,neuronToPlot], embeddingsSTDs=np.sqrt(embeddingVars[:,:,neuronToPlot]), title=title)
    fig.write_image(embeddingsFigFilenamePattern.format("png"))
    fig.write_html(embeddingsFigFilenamePattern.format("html"))

    fig = plot.svGPFA.plotUtilsPlotly.getPlotEmbeddingParams(C=C, d=d)
    fig.write_image(embeddingParamsFigFilenamePattern.format("png"))
    fig.write_html(embeddingParamsFigFilenamePattern.format("html"))

    fig = plot.svGPFA.plotUtilsPlotly.getPlotKernelsParams(
        kernelsTypes=kernelsTypes, kernelsParams=kernelsParams)
    fig.write_image(kernelsParamsFigFilenamePattern.format("png"))
    fig.write_html(kernelsParamsFigFilenamePattern.format("html"))

#     pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
