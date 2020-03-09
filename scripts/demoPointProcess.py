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
from utils.svGPFA.configUtils import getKernels, getLatentsMeansFuncs, getLinearEmbeddingParams

def main(argv):
    if len(argv)!=4:
        print("Usage {:s} <simulation number> <estimation number> <trial to plot>".format(argv[0]))
        return

    # load data and initial values
    simResNumber = int(argv[1])
    estNumber = int(argv[2])
    trialToPlot = int(argv[3])

    simResConfigFilename = "results/{:08d}_simRes.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simConfigFilename = simResConfig["simulation_params"]["simConfiFilename"]
    simResFilename = simResConfig["simulation_results"]["simResFilename"]
    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikeTimes = simRes["spikes"]

    estConfigFilename = "data/{:08d}_estimation_metaData.ini".format(estNumber)
    estConfig = configparser.ConfigParser()
    nIndPointsPerLatent = torch.DoubleTensor([float(str) for str in estConfig["control_variables"]["nIndPointsPerLatent"][1:-1].split(",")])
    nTestPoints = float(estConfig["control_variables"]["nTestPoints"])
    firstIndPoint = float(estConfig["control_variables"]["firstIndPoint"])
    initCondNoiseSTD = float(estConfig["control_variables"]["initCondNoiseSTD"])

#     for i in range(len(spikeTimes)):
#         for j in range(len(spikeTimes[i])):
#             spikeTimes[i][j] = torch.stack(spikeTimes[i][j])
    testTimes = torch.linspace(0, torch.max(spikeTimes[0][0]), nTestPoints)

    simConfig = configparser.ConfigParser()
    simConfig.read(simConfigFilename)
    nLatents = int(simConfig["latents_params"]["nLatents"])
    nTrials = int(simConfig["spikes_params"]["nTrials"])
    trialLengths = torch.IntTensor([int(str) for str in simConfig["simulation_params"]["trialLengths"][1:-1].split(",")])

    C, d = getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, config=simConfig)

    C0 = C + torch.randn(C.shape)*initCondNoiseSTD
    d0 = d + torch.randn(d.shape)*initCondNoiseSTD
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    kernelNames = mat["kernelNames"]
    hprs0 = mat["hprs0"]
    # testTimes = torch.from_numpy(mat['testTimes']).type(torch.DoubleTensor).squeeze()
    indPointsLocsKMSEpsilon = 1e-2

    # create initial conditions for svPosteriorOnIndPoints
    qMu0 = [None]*nLatents
    qSVec0 = [None]*nLatents
    qSDiag0 = [None]*nLatents
    for i in range(nLatents):
        qMu0[i] = torch.zeros(nTrials, nIndPointsPerLatent[i], 1, dtype=torch.double)
        qSVec0[i] = initVariance*torch.eye(nIndPointsPerLatent[i], 1, dtype=torch.double).repeat(nTrials, 1, 1)
        qSDiag0[i] = initVariance*torch.ones(nIndPointsPerLatent[i], 1, dtype=torch.double).repeat(nTrials, 1, 1)

    # create inducing points
    Z0 = [None]*nLatents
    for i in range(nLatents):
        for j in range(nTrials):
            Z0[i] = torch.empty((nTrials, nIndPointsPerLatent[i], 1), dtype=torch.double)
    for i in range(nLatents):
        for j in range(nTrials):
            Z0[i][j,:,0] = torch.linspace(firstIndPoint, trialLengths[j], nIndPointsPerLatent[i])

    pdb.set_trace()

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
    # optimParams = {"emMaxNIter":20, "eStepMaxNIter":100, "mStepModelParamsMaxNIter":100, "mStepKernelParamsMaxNIter":100, "mStepKernelParamsLR":1e-5, "mStepIndPointsMaxNIter":100}
    optimParams = {"emMaxNIter":30, "eStepMaxNIter":100, "mStepModelParamsMaxNIter":100, "mStepKernelParamsMaxNIter":20, "mStepIndPointsMaxNIter":10, "mStepIndPointsLR": 1e-2}

    # create model
    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels,
        indPointsLocsKMSEpsilon=indPointsLocsKMSEpsilon)

    # maximize lower bound
    svEM = stats.svGPFA.svEM.SVEM()
    lowerBoundHist, elapsedTimeHist  = svEM.maximize(
        model=model, measurements=spikeTimes, initialParams=initialParams,
        quadParams=quadParams, optimParams=optimParams)

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
    estimConfig["estimation_params"] = {"nIndPointsPerLatent": nIndPointsPerLatent}
    estimConfig["initial_params"] = {"initDataFilename": initDataFilename}
    with open(estimMetaDataFilename, "w") as f:
        estimConfig.write(f)

    resultsToSave = {"lowerBoundHist": lowerBoundHist, "elapsedTimeHist": elapsedTimeHist, "model": model}
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
