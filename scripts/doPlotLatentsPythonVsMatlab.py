
import sys
import os
import torch
import pdb
import pickle
import matplotlib.pyplot as plt
from scipy.io import loadmat
sys.path.append(os.path.expanduser("../src"))
import plot.svGPFA.plotUtils

def main(argv):
    if len(argv)!=2:
        print("{:s} <trial>".format(argv[0]))
        sys.exit(0)
    trialToPlot = int(argv[1])

    marker = 'x'
    mSimFilename = "../../matlabCode/scripts/results/pointProcessSimulation.mat"
    mModelSaveFilename = "../../matlabCode/scripts/results/pointProcessEstimationRes.mat"
    # pModelSaveFilename = "results/estimationResLeasSimulation.pickle"
    pModelSaveFilename = "results/37816127_leasSimulation_estimatedModel_cpu.pickle"
    staticFigFilenamePattern = "figures/truePythonMatlabLatentsPointProcess_trial{:d}.png"
    dynamicFigFilenamePattern = "figures/truePythonMatlabLatentsPointProcess_trial{:d}.html"

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

        plot.svGPFA.plotUtils.\
            plotTruePythonAndMatlabLatentsPlotly(tTimes=tTimes,
                                                  tLatents=tLatents,
                                                  pTimes=pTimes,
                                                  pMuK=pTestMuK,
                                                  pVarK=pTestVarK,
                                                  mTimes=mTimes,
                                                  mMuK=mMeanLatents,
                                                  mVarK=mVarLatents,
                                                  trialToPlot=trialToPlot,
                                                  staticFigFilenamePattern=
                                                   staticFigFilenamePattern,
                                                  dynamicFigFilenamePattern=
                                                   dynamicFigFilenamePattern)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
