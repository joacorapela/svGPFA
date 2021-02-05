
import sys
import os
import torch
import pdb
import pickle
import argparse
import configparser
import matplotlib.pyplot as plt
from scipy.io import loadmat
sys.path.append("../src")
import plot.svGPFA.plotUtilsPlotly
import utils.svGPFA.miscUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("pEstNumber", help="Python's estimation number", type=int)
    parser.add_argument("trialToPlot", help="Trial to plot", type=int)
    parser.add_argument("neuronToPlot", help="Neuron to plot", type=int)
    parser.add_argument("--deviceName", help="name of device (cpu or cuda)", default="cpu")
    args = parser.parse_args()
    pEstNumber = args.pEstNumber
    trialToPlot = args.trialToPlot
    neuronToPlot = args.neuronToPlot
    deviceName = args.deviceName

    marker = "x"
    tLabel = "True"
    # pLabelPattern = "$\text{Python} (R^2={:.02f})$"
    # mLabelPattern = "$\text{Matlab} (R^2={:.02f})$"
    pLabelPattern = "Python (R<sup>2</sup>={:.02f})"
    mLabelPattern = "Matlab (R<sup>2</sup>={:.02f})"

    pEstimMetaDataFilename = "results/{:08d}_leasSimulation_estimation_metaData_{:s}.ini".format(pEstNumber, deviceName)
    pEstConfig = configparser.ConfigParser()
    pEstConfig.read(pEstimMetaDataFilename)
    mEstNumber = int(pEstConfig["data"]["mEstNumber"])

    mEstParamsFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessEstimationParams.ini".format(mEstNumber)
    mEstConfig = configparser.ConfigParser()
    mEstConfig.read(mEstParamsFilename)
    mSimNumber = int(mEstConfig["data"]["simulationNumber"])

    mSimFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessSimulation.mat".format(mSimNumber)
    mModelSaveFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat".format(mEstNumber)
    pModelSaveFilename = "results/{:08d}_leasSimulation_estimatedModel_cpu.pickle".format(pEstNumber)
    staticFigFilename = "figures/{:08d}_truePythonMatlabCIFsPointProcess_trial{:d}_neuron{:d}.png".format(pEstNumber, trialToPlot, neuronToPlot)
    dynamicFigFilename = "figures/{:08d}_truePythonMatlabCIFsPointProcess_trial{:d}_neuron{:d}.html".format(pEstNumber, trialToPlot, neuronToPlot)

    loadRes = loadmat(mSimFilename)
    nLatents = loadRes["trueLatents"].shape[1]
    nTrials = loadRes["trueLatents"].shape[0]
    nSamples = loadRes["testTimes"][:,0].shape[0]
    tTimes = torch.from_numpy(loadRes["testTimes"][:,0]).type(torch.DoubleTensor)
    tLatents_tmp = [[torch.from_numpy(loadRes["trueLatents"][t,l]).type(torch.DoubleTensor).squeeze() for l in range(nLatents)] for t in range(nTrials)]
    tLatents = torch.empty((nTrials, nSamples, nLatents), dtype=torch.double)
    for t in range(nTrials):
        for l in range(nLatents):
            tLatents[t,:,l] = tLatents_tmp[t][l]

    tC = torch.from_numpy(loadRes['prs'][0,0][0]).type(torch.DoubleTensor)
    td = torch.from_numpy(loadRes['prs'][0,0][1]).type(torch.DoubleTensor)
    tCIFs = utils.svGPFA.miscUtils.getCIFs(C=tC, d=td, latents=tLatents)

    loadRes = loadmat(mModelSaveFilename)
    mTimes = torch.from_numpy(loadRes["testTimes"][:,0]).type(torch.DoubleTensor).squeeze()
    mMeanLatents_tmp = torch.from_numpy(loadRes["meanEstimatedLatents"]).type(torch.DoubleTensor)
    mMeanLatents = torch.empty((nTrials, nSamples, nLatents), dtype=torch.double)
    for t in range(nTrials):
        for l in range(nLatents):
            mMeanLatents[t,:,l] = mMeanLatents_tmp[:,l,t]
    mVarLatents_tmp = torch.from_numpy(loadRes["varEstimatedLatents"]).type(torch.DoubleTensor)
    mVarLatents = torch.empty((nTrials, nSamples, nLatents), dtype=torch.double)
    for t in range(nTrials):
        for l in range(nLatents):
            mVarLatents[t,:,l] = mVarLatents_tmp[:,l,t]
    mC = torch.from_numpy(loadRes["m"]['prs'][0,0]["C"][0,0]).type(torch.DoubleTensor)
    md = torch.from_numpy(loadRes["m"]['prs'][0,0]["b"][0,0]).type(torch.DoubleTensor)
    mCIFs = utils.svGPFA.miscUtils.getCIFs(C=mC, d=md, latents=mMeanLatents)

    with open(pModelSaveFilename, "rb") as f: res = pickle.load(f)
    pModel = res["model"]
    embeddingParams = pModel.getSVEmbeddingParams()
    pC = embeddingParams[0]
    pd = torch.unsqueeze(embeddingParams[1], 1)

    with torch.no_grad():
        pTestMuK, _ = pModel.predictLatents(newTimes=mTimes)
        pCIFs = utils.svGPFA.miscUtils.getCIFs(C=pC, d=pd, latents=pTestMuK)
        pTimes = mTimes

        tCIF = tCIFs[trialToPlot,:,neuronToPlot]
        mCIF = mCIFs[trialToPlot,:,neuronToPlot]
        pCIF = pCIFs[trialToPlot,:,neuronToPlot]
        meanTCIF = torch.mean(tCIF)
        ssTot = torch.sum((tCIF-meanTCIF)**2)
        pSSRes = torch.sum((pCIF-tCIF)**2)
        mSSRes = torch.sum((mCIF-tCIF)**2)
        pR2 = (1-(pSSRes/ssTot)).item()
        mR2 = (1-(mSSRes/ssTot)).item()
        pLabel = pLabelPattern.format(pR2)
        mLabel = mLabelPattern.format(mR2)
        fig = plot.svGPFA.plotUtilsPlotly.\
            getPlotTruePythonAndMatlabCIFsPlotly(tTimes=tTimes,
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
    fig.write_image(staticFigFilename)
    fig.write_html(dynamicFigFilename)
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
