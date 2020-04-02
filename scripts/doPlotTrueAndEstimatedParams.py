
import sys
import os
import pdb
import pickle
import configparser
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly
import plotly.tools as tls
sys.path.append("../src")
from utils.svGPFA.configUtils import getKernels, getLatentsMeansFuncs, getLinearEmbeddingParams
from utils.svGPFA.miscUtils import getLatentsMeanFuncsSamples, getTrialsTimes

def plotTrueAndEstimatedEmbeddingParams(trueC, trueD, estimatedC, estimatedD,
                                        linestyleTrue="-",
                                        linestyleEstimated=":",
                                        xlabel="Neuron Index",
                                        ylabel="Coefficient Value"):
    for i in range(trueC.shape[1]):
        plt.plot(trueC[:,i], label="true C[{:d}]".format(i), linestyle=linestyleTrue)
        plt.plot(estimatedC[:,i], label="est. C[{:d}]".format(i), linestyle=linestyleEstimated)
    plt.plot(trueD, label="true d")
    plt.plot(estimatedD, label="est. d")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

def plotTrueAndEstimatedLatentsMeans(trueLatentsMeans, estimatedLatentsMeans,
                                     trialsTimes,
                                     labelTrue="True",
                                     labelEstimated="Estimated",
                                     xlabel="Time (sec)",
                                     ylabel="Latent Value"):
    def plotOneSetTrueAndEstimatedLatentsMeans(ax, trueLatentMean,
                                               estimatedLatentMean,
                                               times,
                                               labelTrue, labelEstimated,
                                               xlabel, ylabel, useLegend):
            ax.plot(times, trueLatentMean, label=labelTrue)
            ax.plot(times, estimatedLatentMean, label=labelEstimated)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if useLegend:
                ax.legend()

    # trueLatentsMeans[r] \in nLatents x nInd[k]
    # qMu[k] \in nTrials x nInd[k] x 1
    nTrials = len(trueLatentsMeans)
    nLatents = trueLatentsMeans[0].shape[0]
    fig, axs = plt.subplots(nTrials, nLatents)
    for r in range(nTrials):
        times = trialsTimes[r]
        for k in range(nLatents):
            trueLatentMean = trueLatentsMeans[r][k,:]
            estimatedLatentMean = estimatedLatentsMeans[r,:,k]
            if r==0 and k==nLatents-1:
                useLegend = True
            else:
                useLegend = False
            if r==nTrials//2 and k==0:
                ylabelToPlot = ylabel
            else:
                ylabelToPlot = None
            if r==nTrials-1 and k==nLatents//2:
                xlabelToPlot = xlabel
            else:
                xlabelToPlot = None
            plotOneSetTrueAndEstimatedLatentsMeans(ax=axs[r,k],
                                                   trueLatentMean=trueLatentMean,
                                                   estimatedLatentMean=estimatedLatentMean,
                                                   times=times,
                                                   labelTrue=labelTrue,
                                                   labelEstimated=
                                                    labelEstimated,
                                                   xlabel=xlabelToPlot,
                                                   ylabel=ylabelToPlot,
                                                   useLegend=useLegend)

def plotTrueAndEstimatedKernelsParams(trueKernels, estimatedKernelsParams):
    def plotOneSetTrueAndEstimatedKernelsParams(ax, labels,
                                                trueParams,
                                                estimatedParams,
                                                trueLegend = "True",
                                                estimatedLegend = "True",
                                                yLabel="Parameter Value",
                                                useLegend=False):
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        rects1 = ax.bar(x - width/2, trueParams, width, label=trueLegend)
        rects2 = ax.bar(x + width/2, estimatedParams, width, label=estimatedLegend)

        ax.set_ylabel(yLabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        if useLegend:
            ax.legend()

    fig, axs = plt.subplots(len(trueKernels), 1)
    for k in range(len(trueKernels)):
        namedParams = trueKernels[k].getNamedParams()
        labels = namedParams.keys()
        trueParams = [z.item() for z in list(namedParams.values())]
        estimatedParams = estimatedKernelsParams[k].tolist()
        # we are fixing scale to 1.0. This is not great :(
        estimatedParams = [1.0] + estimatedParams
        if k==0:
            useLegend = True
        else:
            useLegend = False
        plotOneSetTrueAndEstimatedKernelsParams(ax=axs[k], labels=labels,
                                                trueParams=trueParams,
                                                estimatedParams=
                                                 estimatedParams,
                                                useLegend=useLegend)

def main(argv):
    if len(argv)!=2:
        print("Usage {:s} <estimation number> ".format(argv[0]))
        return

    estNumber = int(argv[1])
    gpRegularization = 1e-3
    estMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(estNumber)
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estNumber)
    kernelsParamsFigFilename = "figures/true{:08d}AndEstimated{:08d}KernelsParams.png"
    latentsMeansFigFilename = "figures/true{:08d}AndEstimated{:08d}LatentsMeans.png"
    embeddingParamsFigFilename = "figures/true{:08d}AndEstimated{:08d}EmbeddingParams.png"

    estMetaDataConfig = configparser.ConfigParser()
    estMetaDataConfig.read(estMetaDataFilename)
    simResNumber = int(estMetaDataConfig["simulation_params"]["simResNumber"])
    simMetaDataFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simMetaDataConfig = configparser.ConfigParser()
    simMetaDataConfig.read(simMetaDataFilename)
    simConfigFilename = simMetaDataConfig["simulation_params"]["simConfigFilename"]
    simConfig = configparser.ConfigParser()
    simConfig.read(simConfigFilename)
    nLatents = int(simConfig["control_variables"]["nLatents"])
    nNeurons = int(simConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    dtSimulate = float(simConfig["control_variables"]["dt"])

    kernels = getKernels(nLatents=nLatents, nTrials=nTrials, config=simConfig)[0]
    # latentsMeansFuncs[r][k] \in lambda(t)
    tLatentsMeansFuncs = getLatentsMeansFuncs(nLatents=nLatents, nTrials=nTrials, config=simConfig)
    trueC, trueD = getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, config=simConfig)
    trialsTimes = getTrialsTimes(trialsLengths=trialsLengths, dt=dtSimulate)

    # latentsMeansSamples[r][k,t]
    tLatentsMeans = getLatentsMeanFuncsSamples(latentsMeansFuncs=
                                                tLatentsMeansFuncs,
                                               trialsTimes=trialsTimes,
                                               dtype=trueC.dtype)

    with open(modelSaveFilename, "rb") as f: savedResults = pickle.load(f)
    model = savedResults["model"]
    kernelsParams = model.getKernelsParams()
    with torch.no_grad():
        latentsMeans, _ = model.predictLatents(newTimes=trialsTimes[0])
        estimatedC, estimatedD = model.getSVEmbeddingParams()
        # plotTrueAndEstimatedKernelsParams(trueKernels=kernels, estimatedKernelsParams=kernelsParams)
        # plt.savefig(kernelsParamsFigFilename)

        # qMu[r] \in nTrials x nInd[k] x 1
        # plotTrueAndEstimatedLatentsMeans(trueLatentsMeans=tLatentsMeans, estimatedLatentsMeans=latentsMeans, trialsTimes=trialsTimes)
        # plt.savefig(latentsMeansFigFilename)

        plotTrueAndEstimatedEmbeddingParams(trueC=trueC, trueD=trueD, estimatedC=estimatedC, estimatedD=estimatedD)
        # f = plt.gcf()
        # plotly_fig = tls.mpl_to_plotly(f)
        # plotly.offline.plot(plotly_fig, filename="/tmp/tmp.html")
        # plt.savefig(embeddingParamsFigFilename)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
