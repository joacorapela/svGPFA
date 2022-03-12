
import sys
import os
import numpy as np
import torch
import pdb
import pickle
import argparse
import configparser
from scipy.io import loadmat
sys.path.append("../src")
import plot.svGPFA.plotUtilsPlotly
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mEstNumber", help="Matlab's estimation number", type=int)
    parser.add_argument("--trialToPlot", help="Trial to plot", type=int, default=0)
    parser.add_argument("--percent", help="Coverage percent", type=float, default=0.95)
    args = parser.parse_args()
    mEstNumber = args.mEstNumber
    trialToPlot = args.trialToPlot
    percent = args.percent

    mEstParamsFilename = "../../matlabCode/working/scripts/results/{:08d}-pointProcessEstimationParams.ini".format(mEstNumber)
    mModelSaveFilename = "../../matlabCode/working/scripts/results/{:08d}-pointProcessEstimationRes.mat".format(mEstNumber)

    mEstConfig = configparser.ConfigParser()
    mEstConfig.read(mEstParamsFilename)
    pEstResNumber = int(mEstConfig["data"]["pEstNumber"])

    pModelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(pEstResNumber)
    figFilenamePattern = "figures/{:08d}_{:08d}_truePythonAndMatlabEmbeddingPropCovered_trial{:d}.{{:s}}".format(pEstResNumber, mEstNumber, trialToPlot)

    pEstimMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(pEstResNumber)
    pEstConfig = configparser.ConfigParser()
    pEstConfig.read(pEstimMetaDataFilename)
    pSimNumber = int(pEstConfig["simulation_params"]["simResNumber"])

    pSimResMetaDataFilename = "results/{:08d}_simulation_metaData.ini".format(pSimNumber)
    pSimResMetaDataConfig = configparser.ConfigParser()
    pSimResMetaDataConfig.read(pSimResMetaDataFilename)
    pSimInitConfigFilename = pSimResMetaDataConfig["simulation_params"]["simInitConfigFilename"]

    pSimResMetaDataFilename = "results/{:08d}_simulation_metaData.ini".format(pSimNumber)
    pSimResMetaDataConfig = configparser.ConfigParser()
    pSimResMetaDataConfig.read(pSimResMetaDataFilename)
    pSimInitConfigFilename = pSimResMetaDataConfig["simulation_params"]["simInitConfigFilename"]
    pSimResFilename = pSimResMetaDataConfig["simulation_results"]["simResFilename"]

    pSimInitConfig = configparser.ConfigParser()
    pSimInitConfig.read(pSimInitConfigFilename)
    tCFilename = pSimInitConfig["embedding_params"]["C_filename"]
    tDFilename = pSimInitConfig["embedding_params"]["d_filename"]
    tC, td = utils.svGPFA.configUtils.getLinearEmbeddingParams(CFilename=tCFilename, dFilename=tDFilename)
    trialsLengths = [float(str) for str in pSimInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)

    with open(pSimResFilename, "rb") as f: simRes = pickle.load(f)
    tTimes = simRes["latentsTrialsTimes"]
    # tLatentsSamples[r], tLatentsMeans[r], tLatentsVars[r] \in nLatents x nSamples
    tLatentsSamples = simRes["latentsSamples"]
    tLatentsMeans = simRes["latentsMeans"]
    tLatentsSTDs = simRes["latentsSTDs"]

    # tEmbeddingSamples[r], tEmbeddingMeans[r], tEmbeddingSTDs[r] \in nNeurons x nSamples
    tEmbeddingSamples = utils.svGPFA.miscUtils.getEmbeddingSamples(C=tC, d=td,
                                                                   latentsSamples=tLatentsSamples)
    tEmbeddingMeans = utils.svGPFA.miscUtils.getEmbeddingMeans(C=tC, d=td,
                                                               latentsMeans=tLatentsMeans)
    tEmbeddingSTDs = utils.svGPFA.miscUtils.getEmbeddingSTDs(C=tC,
                                                             latentsSTDs=tLatentsSTDs)

    pEstimResConfig = configparser.ConfigParser()
    pEstimResConfig.read(pEstimMetaDataFilename)
    pTimes = torch.unsqueeze(torch.ger(torch.ones(nTrials), tTimes[0]), dim=2)
    with open(pModelSaveFilename, "rb") as f: pEstResults = pickle.load(f)
    pModel = pEstResults["model"]
    pEmbeddingMeans, pEmbeddingVars = pModel.computeEmbeddingMeansAndVarsAtTimes(times=pTimes)

    loadRes = loadmat(mModelSaveFilename)
    mLatentsMeans = loadRes["meanEstimatedLatents"]
    mLatentsVars = loadRes["varEstimatedLatents"]
    if mLatentsMeans.ndim == 3:
        mLatentsMeans = mLatentsMeans.transpose((2, 0, 1))
        mLatentsVars = mLatentsVars.transpose((2, 0, 1))
    elif mLatentsMeans.ndim == 2:
        mLatentsMeans = np.expand_dims(mLatentsMeans, 0)
        mLatentsVars = np.expand_dims(mLatentsVars, 0)
    mTimes = loadRes["latentsTimes"]
    mC = loadRes["m"][0,0]["prs"][0,0]["C"]
    md = loadRes["m"][0,0]["prs"][0,0]["b"]
    mEmbeddingMeans = np.matmul(mLatentsMeans, mC.T) + np.reshape(md, (1, 1, len(md))) # using broadcasting
    mEmbeddingSTDs = np.sqrt(np.matmul(mLatentsVars, (mC.T)**2))

    nNeurons = tEmbeddingSamples[0].shape[0]
    propCovered = np.empty(shape=(3, nNeurons))
    for neuronToPlot in range(nNeurons):
        sample = tEmbeddingSamples[trialToPlot][neuronToPlot,:]

        tMean = tEmbeddingMeans[trialToPlot][neuronToPlot,:]
        tSTD = tEmbeddingSTDs[trialToPlot][neuronToPlot,:]
        propCovered[0, neuronToPlot] = utils.svGPFA.miscUtils.getPropSamplesCovered(sample=sample, mean=tMean, std=tSTD, percent=percent)

        pMean = pEmbeddingMeans[trialToPlot,:,neuronToPlot]
        pSTD = pEmbeddingVars[trialToPlot,:,neuronToPlot].sqrt()
        propCovered[1, neuronToPlot] = utils.svGPFA.miscUtils.getPropSamplesCovered(sample=sample, mean=pMean, std=pSTD, percent=percent)

        mMean = torch.from_numpy(mEmbeddingMeans[trialToPlot,:,neuronToPlot])
        mSTD = torch.from_numpy(mEmbeddingSTDs[trialToPlot,:,neuronToPlot])
        propCovered[2, neuronToPlot] = utils.svGPFA.miscUtils.getPropSamplesCovered(sample=sample, mean=mMean, std=mSTD, percent=percent)
    title = "Trial {:d}".format(trialToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotTruePythonAndMatlabEmbeddingPropCovered(propCovered=propCovered, percent=percent, title=title)

    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
