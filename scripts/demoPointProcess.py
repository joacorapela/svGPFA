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
import myMath.utils
from utils.svGPFA.configUtils import getKernels, getLatentsMeansFuncs, getLinearEmbeddingParams
from stats.pointProcess.tests import KSTestTimeRescalingUnbinned

def getLegQuadPointsAndWeights(nQuad, trialsLengths, dtype=torch.double):
    nTrials = len(trialsLengths)
    legQuadPoints = torch.empty((nTrials, nQuad, 1), dtype=dtype)
    legQuadWeights = torch.empty((nTrials, nQuad, 1), dtype=dtype)
    for r in range(nTrials):
        legQuadPoints[r,:,0], legQuadWeights[r,:,0] = myMath.utils.leggaussVarLimits(n=nQuad, a=0, b=trialsLengths[r])
    return legQuadPoints, legQuadWeights

def getIndPointLocs0(nIndPointsPerLatent, trialsLengths, firstIndPoint):
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
            kernelsParams0[r][k] = noiseSTD*torch.randn(len(trueParams))+trueParams
    return kernelsParams0

def main(argv):
    if len(argv)!=4:
        print("Usage {:s} <simulation result number> <estimation init number> <trial to plot>".format(argv[0]))
        return

    # load data and initial values
    simResNumber = int(argv[1])
    estInitNumber = int(argv[2])
    trialToPlot = int(argv[3])

    simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]
    simResFilename = simResConfig["simulation_results"]["simResFilename"]

    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nLatents = int(simInitConfig["control_variables"]["nLatents"])
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    trueLatents = simRes["latents"]
    trueLatentsMeans = simRes["latentsMeans"]
    trueLatentsSTDs = simRes["latentsSTDs"]
    timesTrueValues = torch.linspace(0, torch.max(torch.tensor(trialsLengths)), trueLatents[0].shape[1])

    estInitConfigFilename = "data/{:08d}_estimation_metaData.ini".format(estInitNumber)
    estInitConfig = configparser.ConfigParser()
    estInitConfig.read(estInitConfigFilename)
    nIndPointsPerLatent = [int(str) for str in estInitConfig["control_variables"]["nIndPointsPerLatent"][1:-1].split(",")]
    nTestPoints = int(estInitConfig["control_variables"]["nTestPoints"])
    firstIndPoint = float(estInitConfig["control_variables"]["firstIndPoint"])
    initCondEmbeddingSTD = float(estInitConfig["control_variables"]["initCondEmbeddingSTD"])
    initCondIndPointsScale = float(estInitConfig["control_variables"]["initCondIndPointsScale"])
    kernelsParams0NoiseSTD = float(estInitConfig["control_variables"]["kernelsParams0NoiseSTD"])
    indPointsLocsKMSRegEpsilon = float(estInitConfig["control_variables"]["indPointsLocsKMSRegEpsilon"])
    nQuad = int(estInitConfig["control_variables"]["nQuad"])

    testTimes = torch.linspace(0, torch.max(torch.tensor(spikesTimes[0][0])), nTestPoints)

    C, d = getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, config=simInitConfig)
    C0 = C + torch.randn(C.shape)*initCondEmbeddingSTD
    d0 = d + torch.randn(d.shape)*initCondEmbeddingSTD

    legQuadPoints, legQuadWeights = getLegQuadPointsAndWeights(nQuad=nQuad, trialsLengths=trialsLengths)

    kernels = getKernels(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
    kernelsParams0 = getKernelsParams0(kernels=kernels, noiseSTD=kernelsParams0NoiseSTD)
    kernels = kernels[0] # the current code uses the same kernel for all trials
    kernelsParams0 = kernelsParams0[0] # the current code uses the same kernel for all trials

    qMu0, qSVec0, qSDiag0 = getSVPosteriorOnIndPointsParams0(nIndPointsPerLatent=nIndPointsPerLatent, nLatents=nLatents, nTrials=nTrials, scale=initCondIndPointsScale)

    Z0 = getIndPointLocs0(nIndPointsPerLatent=nIndPointsPerLatent,
                          trialsLengths=trialsLengths, firstIndPoint=firstIndPoint)

    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    qHParams0 = {"C0": C0, "d0": d0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    initialParams = {"svPosteriorOnIndPoints": qUParams0,
                     "kernelsMatricesStore": kmsParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}
    # optimParams = {"emMaxNIter":20, "eStepMaxNIter":100, "mStepModelParamsMaxNIter":100, "mStepKernelParamsMaxNIter":100, "mStepKernelParamsLR":1e-5, "mStepIndPointsMaxNIter":100}
    # optimParams = {"emMaxNIter":10, "eStepMaxNIter":100, "mStepModelParamsMaxNIter":100, "mStepKernelParamsMaxNIter":20, "mStepIndPointsMaxNIter":10, "mStepIndPointsLR": 1e-2}
    optimParams = {"emMaxNIter":20,
                   #
                   "eStepMaxNIter":100,
                   "eStepTol":1e-3,
                   "eStepLR":1e-3,
                   "eStepLineSearchFn":"strong_wolfe",
                   "eStepNIterDisplay":1,
                   #
                   "mStepModelParamsMaxNIter":100,
                   "mStepModelParamsTol":1e-3,
                   "mStepModelParamsLR":1e-3,
                   "mStepModelParamsLineSearchFn":"strong_wolfe",
                   "mStepModelParamsNIterDisplay":1,
                   #
                   "mStepKernelParamsMaxNIter":10,
                   "mStepKernelParamsTol":1e-3,
                   "mStepKernelParamsLR":1e-3,
                   "mStepKernelParamsLineSearchFn":"strong_wolfe",
                   "mStepModelParamsNIterDisplay":1,
                   "mStepKernelParamsNIterDisplay":1,
                   #
                   "mStepIndPointsMaxNIter":20,
                   "mStepIndPointsTol":1e-3,
                   "mStepIndPointsLR":1e-4,
                   "mStepIndPointsLineSearchFn":"strong_wolfe",
                   "mStepIndPointsNIterDisplay":1}

    # create model
    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels,
        indPointsLocsKMSEpsilon=indPointsLocsKMSRegEpsilon)

    # maximize lower bound
    svEM = stats.svGPFA.svEM.SVEM()
    lowerBoundHist, elapsedTimeHist  = svEM.maximize(
        model=model, measurements=spikesTimes, initialParams=initialParams,
        quadParams=quadParams, optimParams=optimParams)

    # save estimated values
    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estimResMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(estResNumber)
        if not os.path.exists(metaDataFilename):
           estPrefixUsed = False
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)
    latentsFigFilename = "figures/{:08d}_estimatedLatents.png".format(estResNumber)
    lowerBoundHistFigFilename = "figures/{:08d}_lowerBoundHist.png".format(estResNumber)
    ksTestTimeRescalingFigFilename = "figures/{:08d}_ksTestTimeRescaling.png".format(estResNumber)

    estimResConfig = configparser.ConfigParser()
    estimResConfig["simulation_params"] = {"simResNumber": simResNumber}
    estimResConfig["optim_params"] = optimParams
    estimResConfig["estimation_params"] = {"estInitNumber": estInitNumber, "nIndPointsPerLatent": nIndPointsPerLatent}
    with open(estimResMetaDataFilename, "w") as f: estimResConfig.write(f)

    resultsToSave = {"lowerBoundHist": lowerBoundHist, "elapsedTimeHist": elapsedTimeHist, "model": model}
    with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)

    with torch.no_grad():
        # plot lower bound history
        plot.svGPFA.plotUtils.plotLowerBoundHist(lowerBoundHist=lowerBoundHist, figFilename=lowerBoundHistFigFilename)

        # predict latents at test times
        testMuK, testVarK = model.predictLatents(newTimes=testTimes)

        # plot true and estimated latents
        indPointsLocs = model.getIndPointsLocs()
        plot.svGPFA.plotUtils.plotTrueAndEstimatedLatents(timesEstimatedValues=testTimes, muK=testMuK, varK=testVarK, indPointsLocs=indPointsLocs, timesTrueValues=timesTrueValues, trueLatents=trueLatents, trueLatentsMeans=trueLatentsMeans, trueLatentsSTDs=trueLatentsSTDs, trialToPlot=trialToPlot, figFilename=latentsFigFilename)

        # KS test time rescaling
        trialKSTestTimeRescaling = 0
        neuronKSTestTimeRescaling = 0
        dtCIF = 1e-3
        T = torch.tensor(trialsLengths).max()
        oneTrialCIFTimes = torch.arange(0, T, dtCIF)
        cifTimes = torch.unsqueeze(torch.ger(torch.ones(nTrials), oneTrialCIFTimes), dim=2)
        cifs = model.sampleCIFs(times=cifTimes)
        spikesTimesKS = spikesTimes[trialKSTestTimeRescaling][neuronKSTestTimeRescaling]
        cifKS = cifs[trialKSTestTimeRescaling][neuronKSTestTimeRescaling]
        sUTRISIs, uCDF, cb = KSTestTimeRescalingUnbinned(spikesTimes=spikesTimesKS, cif=cifKS, t0=0, tf=T, dt=dtCIF)
        title = "Trial {:d}, Neuron {:d}".format(trialKSTestTimeRescaling, neuronKSTestTimeRescaling)
        plot.svGPFA.plotUtils.plotResKSTestTimeRescaling(sUTRISIs=sUTRISIs, uCDF=uCDF, cb=cb, figFilename=ksTestTimeRescalingFigFilename, title=title)
        pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
