
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

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mEstNumber", help="Matlab's estimation number", type=int)
    parser.add_argument("--trialToPlot", help="Trial to plot", type=int, default=0)
    parser.add_argument("--neuronToPlot", help="Trial to plot", type=int, default=0)
    args = parser.parse_args()
    mEstNumber = args.mEstNumber
    trialToPlot = args.trialToPlot
    neuronToPlot = args.neuronToPlot

    mEstParamsFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessEstimationParams.ini".format(mEstNumber)
    mModelSaveFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat".format(mEstNumber)

    mEstConfig = configparser.ConfigParser()
    mEstConfig.read(mEstParamsFilename)
    pEstResNumber = int(mEstConfig["data"]["pEstNumber"])

    pModelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(pEstResNumber)
    figFilenamePattern = "figures/{:08d}_{:08d}_truePythonAndMatlabEmbedding_trial{:d}_neuron{:d}.{{:s}}".format(pEstResNumber, mEstNumber, trialToPlot, neuronToPlot)

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
    tTimes = simRes["times"]
    # tLatentsSamples[r], tLatentsMeans[r], tLatentsVars[r] \in nLatents x nSamples
    tLatentsSamples = simRes["latents"]
    tLatentsMeans = simRes["latentsMeans"]
    tLatentsSTDs = simRes["latentsSTDs"]

    # tEmbeddingSamples[r], tEmbeddingMeans[r], tEmbeddingSTDs \in nNeurons x nSamples
    tEmbeddingSamples = [torch.matmul(tC, tLatentsSamples[r])+td for r in range(nTrials)]
    tEmbeddingMeans = [torch.matmul(tC, tLatentsMeans[r])+td for r in range(nTrials)]
    tEmbeddingSTDs = [torch.matmul(tC, tLatentsSTDs[r]) for r in range(nTrials)]

    pEstimResConfig = configparser.ConfigParser()
    pEstimResConfig.read(pEstimMetaDataFilename)
    pTimes = torch.unsqueeze(torch.ger(torch.ones(nTrials), tTimes[0]), dim=2)
    with open(pModelSaveFilename, "rb") as f: pEstResults = pickle.load(f)
    pModel = pEstResults["model"]
    pEmbeddingMeans, pEmbeddingVars = pModel.computeEmbeddingMeansAndVarsAtTimes(times=pTimes)

    loadRes = loadmat(mModelSaveFilename)
    mLatentsMeans = loadRes["meanEstimatedLatents"].transpose((2, 0, 1))
    mLatentsVars = loadRes["varEstimatedLatents"].transpose((2, 0, 1))
    mTimes = loadRes["latentsTimes"]
    mC = loadRes["m"][0,0]["prs"][0,0]["C"]
    md = loadRes["m"][0,0]["prs"][0,0]["b"]
    # mEmbeddingMeans = [np.matmul(meC, mELatentsMeans[:,:,r].T)+med for r in range(nTrials)]
    # mEmbeddingSTDs = [np.matmul(meC, mELatentsSTDs[:,:,r].T) for r in range(nTrials)]
    mEmbeddingMeans = np.matmul(mLatentsMeans, mC.T) + np.reshape(md, (1, 1, len(md))) # using broadcasting
    mEmbeddingVars = np.matmul(mLatentsVars, (mC.T)**2)

    tSamplesToPlot = tEmbeddingSamples[trialToPlot][neuronToPlot,:]
    tMeansToPlot = tEmbeddingMeans[trialToPlot][neuronToPlot,:]
    tSTDsToPlot = tEmbeddingSTDs[trialToPlot][neuronToPlot,:]
    pMeansToPlot = pEmbeddingMeans[trialToPlot,:,neuronToPlot]
    pSTDsToPlot = pEmbeddingVars[trialToPlot,:,neuronToPlot].sqrt()
    mMeansToPlot = mEmbeddingMeans[trialToPlot,:,neuronToPlot]
    mSTDsToPlot = np.sqrt(mEmbeddingVars[trialToPlot,:,neuronToPlot])

    title = "Trial {:d}, Neuron {:d}".format(trialToPlot, neuronToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotTruePythonAndMatlabEmbedding(tTimes=tTimes[trialToPlot], tSamples=tSamplesToPlot, tMeans=tMeansToPlot, tSTDs=tSTDsToPlot, pTimes=pTimes[trialToPlot,:,0], pMeans=pMeansToPlot, pSTDs=pSTDsToPlot, mTimes=mTimes[:,0], mMeans=mMeansToPlot, mSTDs=mSTDsToPlot, title=title)

    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
