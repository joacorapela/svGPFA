
import sys
import os
import pdb
import pickle
import argparse
import configparser
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly
import plotly.tools as tls
import scipy.io
sys.path.append("../src")
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import plot.svGPFA.plotUtils
import plot.svGPFA.plotUtilsPlotly

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("pEstNumber", help="Python's estimation number", type=int)
    parser.add_argument("--deviceName", help="name of device (cpu or cuda)", default="cpu")
    args = parser.parse_args()
    pEstNumber = args.pEstNumber
    deviceName = args.deviceName
    gpRegularization = 1e-3

    pEstimMetaDataFilename = "results/{:08d}_leasSimulation_estimation_metaData_{:s}.ini".format(pEstNumber, deviceName)
    pEstConfig = configparser.ConfigParser()
    pEstConfig.read(pEstimMetaDataFilename)
    mEstNumber = int(pEstConfig["data"]["mEstNumber"])

    mEstConfig = configparser.ConfigParser()
    mEstConfig.read("../../matlabCode/scripts/results/{:08d}-pointProcessEstimationParams.ini".format(mEstNumber))
    mSimNumber = int(mEstConfig["data"]["simulationNumber"])
    simDataFilename = os.path.join(os.path.dirname(__file__), "../../matlabCode/scripts/results/{:08d}-pointProcessSimulation.mat".format(mSimNumber))
    simInitialCondsFilename = os.path.join(os.path.dirname(__file__), "../../matlabCode/scripts/results/{:08d}-pointProcessInitialConditions.mat".format(mEstNumber))
    modelSaveFilename = "results/{:08d}_leasSimulation_estimatedModel_{:s}.pickle".format(pEstNumber, "cpu")
    kernelsParamsFigFilenamePattern = "figures/{:08d}_trueAndEstimatedKernelsParams.{{:s}}".format(pEstNumber)
    latentsMeansFigFilenamePattern = "figures/{:08d}_trueAndEstimatedLatentsMeans.{{:s}}".format(pEstNumber)
    embeddingParamsFigFilenamePattern = "figures/{:08d}_trueAndEstimatedEmbeddingParams.{{:s}}".format(pEstNumber)

    simData = scipy.io.loadmat(simDataFilename)

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
    CFilename = simInitConfig["embedding_params"]["C_filename"]
    dFilename = simInitConfig["embedding_params"]["d_filename"]
    trueC, trueD = utils.svGPFA.configUtils.getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, CFilename=CFilename, dFilename=dFilename)
    trialsTimes = utils.svGPFA.miscUtils.getTrialsTimes(trialsLengths=trialsLengths, dt=dtSimulate)

    # latentsMeansSamples[r][k,t]
    tLatentsMeans = utils.svGPFA.miscUtils.getLatentsMeanFuncsSamples(latentsMeansFuncs=tLatentsMeansFuncs, trialsTimes=trialsTimes, dtype=trueC.dtype)

    with open(modelSaveFilename, "rb") as f: savedResults = pickle.load(f)
    model = savedResults["model"]
    kernelsParams = model.getKernelsParams()
    with torch.no_grad():
        latentsMeans, _ = model.predictLatents(newTimes=trialsTimes[0])
        estimatedC, estimatedD = model.getSVEmbeddingParams()

    fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedKernelsParamsPlotly(trueKernels=kernels, estimatedKernelsParams=kernelsParams)
    fig.write_image(kernelsParamsFigFilenamePattern.format("png"))
    fig.write_html(kernelsParamsFigFilenamePattern.format("html"))
    plt.clf()

    # qMu[r] \in nTrials x nInd[k] x 1
    fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedLatentsMeansPlotly(trueLatentsMeans=tLatentsMeans, estimatedLatentsMeans=latentsMeans, trialsTimes=trialsTimes)
    fig.write_image(latentsMeansFigFilenamePattern.format("png"))
    fig.write_html(latentsMeansFigFilenamePattern.format("html"))
    plt.clf()

    fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedEmbeddingParamsPlotly(trueC=trueC, trueD=trueD, estimatedC=estimatedC, estimatedD=estimatedD)
    fig.write_image(embeddingParamsFigFilenamePattern.format("png"))
    fig.write_html(embeddingParamsFigFilenamePattern.format("html"))
    plt.clf()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
