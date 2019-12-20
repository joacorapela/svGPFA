
import sys
import os
import pdb
from scipy.io import loadmat
import torch
import pickle
import configparser
sys.path.append(os.path.expanduser("~/dev/research/programs/src/python"))
import plot.svGPFA.plotUtils
import matplotlib.pyplot as plt

def main(argv):
    if len(argv)!=3:
        print("Usage {:s} <random prefix> <trial to plot>".format(argv[0]))
        return

    randomPrefix = argv[1]
    trialToPlot = int(argv[2])
    eLatentsFigFilename = "figures/{:s}_trial{:d}_estimatedLatents.png".format(randomPrefix, trialToPlot)
    dataFilename = "data/demo_PointProcess.mat"

    modelSaveFilename = \
        "results/{:s}_estimatedModel.pickle".format(randomPrefix)
    lowerBoundHistFigFilename = \
        "figures/{:s}_lowerBoundHist.png".format(randomPrefix)

    estConfigFilename = "results/{:s}_estimation_metaData.ini".format(randomPrefix)
    estConfig = configparser.ConfigParser()
    estConfig.read(estConfigFilename)
    simPrefix = estConfig["simulation_params"]["simprefix"]
    latentsFilename = "results/{:s}_latents.pickle".format(simPrefix)

    with open(modelSaveFilename, "rb") as f: savedResults = pickle.load(f)
    model = savedResults["model"]
    lowerBoundHist = savedResults["lowerBoundHist"]
    plot.svGPFA.plotUtils.plotLowerBoundHist(lowerBoundHist=lowerBoundHist, figFilename=lowerBoundHistFigFilename)

    with open(latentsFilename, "rb") as f: trueLatentsSamples = pickle.load(f)
    mat = loadmat(dataFilename)
    testTimes = torch.from_numpy(mat['testTimes']).type(torch.DoubleTensor).squeeze()

    testMuK, testVarK = model.predictLatents(newTimes=testTimes)
    indPointsLocs = model.getIndPointsLocs()
    plot.svGPFA.plotUtils.plotTrueAndEstimatedLatents(times=testTimes, muK=testMuK, varK=testVarK, indPointsLocs=indPointsLocs, trueLatents=trueLatentsSamples, trialToPlot=trialToPlot, figFilename=eLatentsFigFilename)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
