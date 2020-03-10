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
import myMath.utils

def getLegQuadPointsAndWeights(nQuad, trialsLengths):
    nTrials = len(trialsLengths)
    legQuadPoints = torch.empty((nTrials, nQuad, 1))
    legQuadWeights = torch.empty((nTrials, nQuad, 1))
    for r in range(nTrials):
        leqQuadPoints[r,:,0], leqQuadWeights[r,:,0] = math.utils.leggaussVarLimits(n=nQuad, a=0, b=trialsLengths[r])
    return legQuadPoints, legQuadWeights

def getIndPointLocs0(nIndPointsPerLatent, trialsLengths):
    nLatents = len(nIndPointsPerLatent)
    nTrials = len(trialsLengths)

    Z0 = [None]*nLatents
    for i in range(nLatents):
        Z0[i] = torch.empty((nTrials, nIndPointsPerLatent[i], 1), dtype=torch.double)
    for i in range(nLatents):
        for j in range(nTrials):
            Z0[i][j,:,0] = torch.linspace(firstIndPoint, trialsLengths[j], nIndPointsPerLatent[i])
    return Z0

def getSVPosteriorOnIndPointsParams0(nIndPointsPerLatent, nLatents, nTrials, scale):
    qMu0 = [None]*nLatents
    qSVec0 = [None]*nLatents
    qSDiag0 = [None]*nLatents
    for i in range(nLatents):
        qMu0[i] = torch.zeros(nTrials, nIndPointsPerLatent[i], 1, dtype=torch.double)
        qSVec0[i] = scale*torch.eye(nIndPointsPerLatent[i], 1, dtype=torch.double).repeat(nTrials, 1, 1)
        qSDiag0[i] = scale*torch.ones(nIndPointsPerLatent[i], 1, dtype=torch.double).repeat(nTrials, 1, 1)
    return qMu0, qSVec0, qSDiag0

def getKernelsParams0(kernels, noiseSTD):
    nTrials = len(kernels)
    nLatents = len(kernels[0])
    kernelsParams0 = [[] for r in range(nTrials)]
    for r in range(nTrials):
        kernelsParams0[r] = [[] for r in range(nLatents)]
        for k in range(nLatents):
            trueParams = kernels[r][k].getParams()
            kernelsParams0[r][k] = noiseSTD*torch.randn(len(trueParams))*trueParams
    return kernelsParams0

def main(argv):
    if len(argv)!=4:
        print("Usage {:s} <simulation number> <estimation number> <trial to plot>".format(argv[0]))
        return

    # load data and initial values
    simResNumber = int(argv[1])
    estNumber = int(argv[2])
    trialToPlot = int(argv[3])

    simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simConfigFilename = simResConfig["simulation_params"]["simConfigFilename"]
    simResFilename = simResConfig["simulation_results"]["simResFilename"]
    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikeTimes = simRes["spikes"]

    estConfigFilename = "data/{:08d}_estimation_metaData.ini".format(estNumber)
    estConfig = configparser.ConfigParser()
    estConfig.read(estConfigFilename)
    nIndPointsPerLatent = torch.DoubleTensor([float(str) for str in estConfig["control_variables"]["nIndPointsPerLatent"][1:-1].split(",")])
    nTestPoints = float(estConfig["control_variables"]["nTestPoints"])
    firstIndPoint = float(estConfig["control_variables"]["firstIndPoint"])
    initCondEmbeddingSTD = float(estConfig["control_variables"]["initCondEmbeddingSTD"])
    initCondIndPointsScale = float(estConfig["control_variables"]["initCondIndPointsScale"])
    kernelsParams0NoiseSTD = float(estConfig["control_variables"]["kernelsParams0NoiseSTD"])
    indPointsLocsKMSRegEpsilon = float(estConfig["control_variables"]["indPointsLocsKMSRegEpsilon"])

    testTimes = torch.linspace(0, torch.max(spikeTimes[0][0]), nTestPoints)

    simConfig = configparser.ConfigParser()
    simConfig.read(simConfigFilename)
    nLatents = int(simConfig["latents_params"]["nLatents"])
    nTrials = int(simConfig["spikes_params"]["nTrials"])
    trialsLengths = torch.IntTensor([int(str) for str in simConfig["simulation_params"]["trialsLengths"][1:-1].split(",")])

    C, d = getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, config=simConfig)
    C0 = C + torch.randn(C.shape)*initCondNoiseSTD
    d0 = d + torch.randn(d.shape)*initCondNoiseSTD

    legQuadPoints, legQuadWeights = getLegQuadPointsAndWeights(nQuad=nQuad, trialsLengths=trialsLengths)

    kernels = getKernels(nLatents=nLatents, nTrials=nTrials, config=simConfig)
    kernels = kernels[0] # the current code uses the same kernel for all trials
    kernelsParams0 = getKernelsParams0(kernels=kernels, noiseSTD=kernelsParams0NoiseSTD)
    kernelsParams0 = kernelsParams0[0] # the current code uses the same kernel for all trials

    qMu0, qSVec0, qSDiag0 = getSVPosteriorOnIndPointsParams0(nIndPointsPerLatent=nIndPointsPerLatent, nLatents=nLatents, nTrials=nTrials, scale=initCondIndPointsScale)

    Z0 = getIndPointLocs0(nIndPointsPerLatent=nIndPointsPerLatent, trialsLengths=trialsLengths)

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
    optimParams = {"emMaxNIter":10, "eStepMaxNIter":100, "mStepModelParamsMaxNIter":100, "mStepKernelParamsMaxNIter":20, "mStepIndPointsMaxNIter":10, "mStepIndPointsLR": 1e-2}

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
