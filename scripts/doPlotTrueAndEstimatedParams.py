
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
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import plot.svGPFA.plotUtils

def main(argv):
    if len(argv)!=2:
        print("Usage {:s} <estimation number> ".format(argv[0]))
        return

    estNumber = int(argv[1])
    gpRegularization = 1e-3
    estResConfigFilename = "results/{:08d}_estimation_metaData.ini".format(estNumber)
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estNumber)
    kernelsParamsFigFilename = "figures/{:08d}_trueAndEstimatedKernelsParams.png".format(estNumber)
    latentsMeansFigFilename = "figures/{:08d}_trueAndEstimatedLatentsMeans.png".format(estNumber)
    embeddingParamsFigFilename = "figures/{:08d}_trueAndEstimatedEmbeddingParams.png".format(estNumber)

    estResConfig = configparser.ConfigParser()
    estResConfig.read(estResConfigFilename)
    simResNumber = int(estResConfig["simulation_params"]["simResNumber"])
    simResFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResFilename)
    simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]
    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nLatents = int(simInitConfig["control_variables"]["nLatents"])
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    dtSimulate = float(simInitConfig["control_variables"]["dt"])

    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)[0]
    # latentsMeansFuncs[r][k] \in lambda(t)
    tLatentsMeansFuncs = utils.svGPFA.configUtils.getLatentsMeansFuncs(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
    trueC, trueD = utils.svGPFA.configUtils.getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, config=simInitConfig)
    trialsTimes = utils.svGPFA.miscUtils.getTrialsTimes(trialsLengths=trialsLengths, dt=dtSimulate)

    # latentsMeansSamples[r][k,t]
    tLatentsMeans = utils.svGPFA.miscUtils.getLatentsMeanFuncsSamples(latentsMeansFuncs=
                                                tLatentsMeansFuncs,
                                               trialsTimes=trialsTimes,
                                               dtype=trueC.dtype)

    with open(modelSaveFilename, "rb") as f: savedResults = pickle.load(f)
    model = savedResults["model"]
    kernelsParams = model.getKernelsParams()
    with torch.no_grad():
        latentsMeans, _ = model.predictLatents(newTimes=trialsTimes[0])
        estimatedC, estimatedD = model.getSVEmbeddingParams()

    plot.svGPFA.plotUtils.plotTrueAndEstimatedKernelsParams(trueKernels=kernels, estimatedKernelsParams=kernelsParams)
    plt.savefig(kernelsParamsFigFilename)

    # qMu[r] \in nTrials x nInd[k] x 1
    plot.svGPFA.plotUtils.plotTrueAndEstimatedLatentsMeans(trueLatentsMeans=tLatentsMeans, estimatedLatentsMeans=latentsMeans, trialsTimes=trialsTimes)
    plt.savefig(latentsMeansFigFilename)

    plot.svGPFA.plotUtils.plotTrueAndEstimatedEmbeddingParams(trueC=trueC, trueD=trueD, estimatedC=estimatedC, estimatedD=estimatedD)
    plt.savefig(embeddingParamsFigFilename)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
