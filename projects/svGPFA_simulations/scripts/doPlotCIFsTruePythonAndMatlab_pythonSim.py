
import sys
import os
import torch
import pdb
import pickle
import argparse
import configparser
import scipy.io
import matplotlib.pyplot as plt
from scipy.io import loadmat
sys.path.append("../src")
import plot.svGPFA.plotUtilsPlotly
import utils.svGPFA.miscUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mEstNumber", help="Matlab's estimation number", type=int)
    parser.add_argument("trialToPlot", help="Trial to plot", type=int)
    parser.add_argument("neuronToPlot", help="Neuron to plot", type=int)
    args = parser.parse_args()
    mEstNumber = args.mEstNumber
    trialToPlot = args.trialToPlot
    neuronToPlot = args.neuronToPlot

    marker = "x"
    tLabel = "True"
    ylim = [-6, 2]
    nResamples = 10000
    # pLabelPattern = "$\text{Python} (R^2={:.02f})$"
    # mLabelPattern = "$\text{Matlab} (R^2={:.02f})$"
    pLabelPattern = "Python (R<sup>2</sup>={:.02f})"
    mLabelPattern = "Matlab (R<sup>2</sup>={:.02f})"

    mEstParamsFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessEstimationParams.ini".format(mEstNumber)
    mEstConfig = configparser.ConfigParser()
    mEstConfig.read(mEstParamsFilename)
    pEstNumber = int(mEstConfig["data"]["pEstNumber"])

    pEstimMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(pEstNumber)
    pEstConfig = configparser.ConfigParser()
    pEstConfig.read(pEstimMetaDataFilename)
    pSimNumber = int(pEstConfig["simulation_params"]["simResNumber"])

    pSimResFilename = "results/{:08d}_simRes.pickle".format(pSimNumber)
    mModelSaveFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat".format(mEstNumber)
    pModelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(pEstNumber)
    figFilenamePattern = "figures/{:08d}-{:08d}-truePythonMatlabCIFsPointProcess.{{:s}}".format(mEstNumber, pEstNumber)

    with open(pSimResFilename, "rb") as f: simRes = pickle.load(f)
    nTrials = len(simRes["latents"])
    nLatents = len(simRes["latents"][0])
    tNSamples = len(simRes["times"][0])
    tTimes = simRes["times"][0]
    tLatents = torch.empty((nTrials, tNSamples, nLatents), dtype=torch.double)
    for r in range(nTrials):
        for k in range(nLatents):
            tLatents[r,:,k] = simRes["latents"][r][k]

    tC = simRes["C"]
    nNeurons = tC.shape[0]
    td = simRes["d"]
    tCIFs = utils.svGPFA.miscUtils.getCIFs(C=tC, d=td, latents=tLatents)

    loadRes = scipy.io.loadmat(mModelSaveFilename)
    mTimes = torch.from_numpy(loadRes["testTimes"][:,0]).type(torch.DoubleTensor).squeeze()
    mMeanLatents_tmp = torch.from_numpy(loadRes["meanEstimatedLatents"]).type(torch.DoubleTensor)
    eNSamples = mMeanLatents_tmp.shape[0]
    # mMeanLatents_tmp = torch.reshape(mMeanLatents_tmp, (-1, nTrials, nLatents))
    mMeanLatents = torch.empty((nTrials, eNSamples, nLatents), dtype=torch.double)
    for r in range(nTrials):
        for k in range(nLatents):
            mMeanLatents[r,:,k] = mMeanLatents_tmp[:,k,r]
    mVarLatents_tmp = torch.from_numpy(loadRes["varEstimatedLatents"]).type(torch.DoubleTensor)
    # mVarLatents_tmp = torch.reshape(mVarLatents_tmp, (-1, nTrials, nLatents))
    mVarLatents = torch.empty((nTrials, eNSamples, nLatents), dtype=torch.double)
    for r in range(nTrials):
        for k in range(nLatents):
            mVarLatents[r,:,k] = mVarLatents_tmp[:,k,r]
    mC = torch.from_numpy(loadRes["m"]["prs"][0,0]["C"][0,0]).type(torch.DoubleTensor)
    md = torch.from_numpy(loadRes["m"]["prs"][0,0]["b"][0,0]).type(torch.DoubleTensor)
    mCIFs = utils.svGPFA.miscUtils.getCIFs(C=mC, d=md, latents=mMeanLatents)

    with open(pModelSaveFilename, "rb") as f: res = pickle.load(f)
    pModel = res["model"]
    embeddingParams = pModel.getSVEmbeddingParams()
    pC = embeddingParams[0]
    pd = embeddingParams[1]

    with torch.no_grad():
        pTestMuK, _ = pModel.predictLatents(newTimes=mTimes)
    pCIFs = utils.svGPFA.miscUtils.getCIFs(C=pC, d=pd, latents=pTestMuK)
    pTimes = mTimes

    tCIF = tCIFs[trialToPlot,:,neuronToPlot]
    mCIF = mCIFs[trialToPlot,:,neuronToPlot]
    pCIF = pCIFs[trialToPlot,:,neuronToPlot]
    meanTCIF = torch.mean(tCIF)
    # ssTot = torch.sum((tCIF-meanTCIF)**2)
    # pSSRes = torch.sum((pCIF-tCIF)**2)
    # mSSRes = torch.sum((mCIF-tCIF)**2)
    # pR2 = (1-(pSSRes/ssTot)).item()
    # mR2 = (1-(mSSRes/ssTot)).item()
    # pLabel = pLabelPattern.format(pR2)
    pLabel = "Python"
    # mLabel = mLabelPattern.format(mR2)
    mLabel = "Matlab"
    fig = plot.svGPFA.plotUtilsPlotly.\
        getPlotTruePythonAndMatlabCIFs(tTimes=tTimes,
                                             tCIF=tCIF,
                                             tLabel=tLabel,
                                             pTimes=pTimes,
                                             pCIF=pCIF,
                                             pLabel=pLabel,
                                             mTimes=mTimes,
                                             mCIF=mCIF,
                                             mLabel=mLabel,
                                             title="Trial {:d}, Neuron {:d}".format(trialToPlot, neuronToPlot),
                                            )
    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
