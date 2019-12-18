import sys
import os
import pdb
import random
from scipy.io import loadmat
import torch
import numpy as np
import pickle
import configparser
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser("../src"))
import stats.kernels
import stats.svGPFA.svGPFAModelFactory
import stats.svGPFA.svEM
import plot.svGPFA.plotUtils

def main(argv):
    if len(argv)!=3:
        print("Usage {:s} <random prefix> <trial to plot>".format(argv[0]))
        return

    # load data and initial values
    simPrefix = argv[1]
    trialToPlot = int(argv[2])
    optimParams = {"emMaxNIter":30, "eStepMaxNIter":100, "mStepModelParamsMaxNIter":100, "mStepKernelParamsMaxNIter":100, "mStepKernelParamsLR":1e-5, "mStepIndPointsMaxNIter":100}
    initDataFilename = os.path.join("data/demo_PointProcess.mat")
    spikeTimesFilename = \
        "results/{:s}_spikeTimes.pickle".format(simPrefix)
    with open(spikeTimesFilename, "rb") as f: spikeTimes = pickle.load(f)

    simConfigFilename = "results/{:s}_metaData.ini".format(simPrefix)
    simConfig = configparser.ConfigParser()
    simConfig.read(simConfigFilename)
    nLatents = int(simConfig["latents_params"]["nLatents"])
    nTrials = int(simConfig["spikes_params"]["nTrials"])

    mat = loadmat(initDataFilename)
    qMu0 = [torch.from_numpy(mat['q_mu0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze()
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    kernelNames = mat["kernelNames"]
    hprs0 = mat["hprs0"]
    testTimes = torch.from_numpy(mat['testTimes']).type(torch.DoubleTensor).squeeze()

    # create kernels
    kernels = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = stats.kernels.PeriodicKernel()
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = stats.kernels.ExponentialQuadraticKernel()
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    # create initial parameters
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernelsParams0[k] = torch.tensor([1.0,
                                              float(hprs0[k,0][0]),
                                              float(hprs0[k,0][1])],
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernelsParams0[k] = torch.tensor([1.0,
                                              float(hprs0[k,0][0])],
                                             dtype=torch.double)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    qHParams0 = {"C0": C0, "d0": b0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    initialParams = {"svPosteriorOnIndPoints": qUParams0,
                     "kernelsMatricesStore": kmsParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    # create model
    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels)

    # maximize lower bound
    svEM = stats.svGPFA.svEM.SVEM()
    lowerBoundHist = svEM.maximize(model=model, measurements=spikeTimes,
                                   initialParams=initialParams,
                                   quadParams=quadParams,
                                   optimParams=optimParams)

    # save estimated values
    estimationPrefixUsed = True
    while estimationPrefixUsed:
        estimationPrefix = "{:08d}".format(random.randint(0, 10**8))
        metaDataFilename = \
            "results/{:s}_estimation_metaData.csv".format(estimationPrefix)
        if not os.path.exists(metaDataFilename):
           estimationPrefixUsed = False
    estimMetaDataFilename = \
        "results/{:s}_estimation_metaData.ini".format(estimationPrefix)
    modelSaveFilename = \
        "results/{:s}_estimatedModel.pickle".format(estimationPrefix)
    latentsFilename = "results/{:s}_latents.pickle".format(simPrefix)
    latentsFigFilename = "figures/{:s}_estimatedLatents.png".format(estimationPrefix)
    lowerBoundHistFigFilename = \
        "figures/{:s}_lowerBoundHist.png".format(estimationPrefix)

    estimConfig = configparser.ConfigParser()
    estimConfig["simulation_params"] = {"simPrefix": simPrefix}
    estimConfig["optim_params"] = optimParams
    estimConfig["initial_params"] = {"initDataFilename": initDataFilename}
    with open(estimMetaDataFilename, "w") as f:
        estimConfig.write(f)

    resultsToSave = {"lowerBoundHist": lowerBoundHist, "model": model}
    with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)
    with open(latentsFilename, "rb") as f: trueLatentsSamples = pickle.load( f)

    # plot lower bound history
    plot.svGPFA.plotUtils.plotLowerBoundHist(lowerBoundHist=lowerBoundHist, figFilename=lowerBoundHistFigFilename)

    # predict latents at test times
    testMuK, testVarK = model.predictLatents(newTimes=testTimes)

    # plot true and estimated latents
    plt.figure()
    indPointsLocs = model.getIndPointsLocs()
    plot.svGPFA.plotUtils.plotTrueAndEstimatedLatents(times=testTimes, muK=testMuK, varK=testVarK, indPointsLocs=indPointsLocs, trueLatents=trueLatentsSamples, trialToPlot=trialToPlot, figFilename=latentsFigFilename)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
