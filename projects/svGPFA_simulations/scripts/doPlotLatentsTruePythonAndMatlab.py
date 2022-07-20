
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

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("pEstNumber", help="Python's estimation number", type=int)
    parser.add_argument("trialToPlot", help="Trial to plot", type=int)
    parser.add_argument("--deviceName", help="name of device (cpu or cuda)", default="cpu")
    args = parser.parse_args()
    pEstNumber = args.pEstNumber
    trialToPlot = args.trialToPlot
    deviceName = args.deviceName

    marker = 'x'

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
    staticFigFilename = "figures/{:08d}_truePythonMatlabLatentsPointProcess_trial{:d}.png".format(pEstNumber, trialToPlot)
    dynamicFigFilename = "figures/{:08d}_truePythonMatlabLatentsPointProcess_trial{:d}.html".format(pEstNumber, trialToPlot)

    loadRes = loadmat(mSimFilename)
    nLatents = loadRes["trueLatents"].shape[1]
    nTrials = loadRes["trueLatents"].shape[0]
    nSamples = loadRes["testTimes"][:,0].shape[0]
    tTimes = torch.from_numpy(loadRes["testTimes"][:,0]).type(torch.DoubleTensor)
    tLatents_tmp = [[torch.from_numpy(loadRes["trueLatents"][t,l]).type(torch.DoubleTensor).squeeze() for l in range(nLatents)] for t in range(nTrials)]
    tLatents = torch.empty((nTrials, nSamples, nLatents))
    for t in range(nTrials):
        for l in range(nLatents):
            tLatents[t,:,l] = tLatents_tmp[t][l]

    loadRes = loadmat(mModelSaveFilename)
    mTimes = torch.from_numpy(loadRes["testTimes"][:,0]).type(torch.DoubleTensor).squeeze()
    mMeanLatents_tmp = torch.from_numpy(loadRes["meanEstimatedLatents"]).type(torch.DoubleTensor)
    mMeanLatents = torch.empty((nTrials, nSamples, nLatents))
    for t in range(nTrials):
        for l in range(nLatents):
            mMeanLatents[t,:,l] = mMeanLatents_tmp[:,l,t]
    mVarLatents_tmp = torch.from_numpy(loadRes["varEstimatedLatents"]).type(torch.DoubleTensor)
    mVarLatents = torch.empty((nTrials, nSamples, nLatents))
    for t in range(nTrials):
        for l in range(nLatents):
            mVarLatents[t,:,l] = mVarLatents_tmp[:,l,t]

    with open(pModelSaveFilename, "rb") as f: res = pickle.load(f)
    pModel = res["model"]

    with torch.no_grad():
        pTestMuK, pTestVarK = pModel.predictLatents(newTimes=mTimes)
        pTimes = mTimes

        fig = plot.svGPFA.plotUtilsPlotly.\
            getPlotTruePythonAndMatlabLatentsPlotly(tTimes=tTimes,
                                                    tLatents=tLatents,
                                                    pTimes=pTimes,
                                                    pMuK=pTestMuK,
                                                    pVarK=pTestVarK,
                                                    mTimes=mTimes,
                                                    mMuK=mMeanLatents,
                                                    mVarK=mVarLatents,
                                                    trialToPlot=trialToPlot,
                                                   )
    fig.write_image(staticFigFilename)
    fig.write_html(dynamicFigFilename)
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
