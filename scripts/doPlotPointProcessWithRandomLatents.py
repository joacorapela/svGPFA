
import sys
import os
import pdb
from scipy.io import loadmat
import torch
import pickle
import plotUtils

def main(argv):
    k0Scale, k0LengthScale, k0Period = 0.1, 1.5, 1/2.5
    k1Scale, k1LengthScale, k1Period =.1, 1.2, 1/2.5,
    k2Scale, k2LengthScale = .1, 1
    trialToPlot = 0
    modelSaveFilename = os.path.join(os.path.dirname(__file__),
                                     "results/estimatedsvGPFAModel.pickle")
    latentsFilename = "results/latents_k0Scale{:.2f}_k0LengthScale{:.2f}_k0Period{:.2f}_k1Scale{:.2f}_k1LengthScale{:.2f}_k1Period{:.2f}_k2Scale{:.2f}_k2LengthScale{:.2f}.pickle".format(k0Scale, k0LengthScale, k0Period, k1Scale, k1LengthScale, k1Period, k2Scale, k2LengthScale)
    dataFilename = os.path.join(os.path.dirname(__file__),
                                "data/demo_PointProcess.mat")
    figFilename = os.path.join(os.path.dirname(__file__),
                               "figures/estimatedLatents.png")

    with open(modelSaveFilename, "rb") as f: model = pickle.load(f)
    with open(latentsFilename, "rb") as f: trueLatentsSamples = pickle.load(f)

    mat = loadmat(dataFilename)
    testTimes = torch.from_numpy(mat['testTimes']).type(torch.DoubleTensor).squeeze()

    testMuK, testVarK = model.predictLatents(newTimes=testTimes)
    indPointsLocs = model.getIndPointsLocs()
    plotUtils.plotTrueAndEstimatedLatents(times=testTimes, muK=testMuK, varK=testVarK, indPointsLocs=indPointsLocs, trueLatents=trueLatentsSamples, trialToPlot=trialToPlot, figFilename=figFilename)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
