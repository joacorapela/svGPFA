import sys
import os
import pdb
import random
import scipy.io
import numpy as np
import torch
import pickle
import argparse
import configparser
import scipy.io
sys.path.append("../src")
import stats.kernels
import stats.svGPFA.svGPFAModelFactory
import stats.svGPFA.svEM
import myMath.utils
import utils.svGPFA.configUtils
import stats.pointProcess.tests
import utils.svGPFA.miscUtils
import utils.svGPFA.initUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simResNumber", help="Simuluation result number", type=int)
    parser.add_argument("estInitNumber", help="estimation init number", type=int)
    parser.add_argument("--nLatents", help="number of latents to use in the estimation", type=int, default=-1)
    args = parser.parse_args()

    simResNumber = args.simResNumber
    estInitNumber = args.estInitNumber
    nLatents = args.nLatents

    # load data and initial values
    simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]
    simResFilename = simResConfig["simulation_results"]["simResFilename"]

    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    if nLatents<0:
        nLatents = int(simInitConfig["control_variables"]["nLatents"])
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]

    estInitConfigFilename = "data/{:08d}_estimation_metaData.ini".format(estInitNumber)
    estInitConfig = configparser.ConfigParser()
    estInitConfig.read(estInitConfigFilename)
    nIndPointsPerLatent = [int(str) for str in estInitConfig["control_variables"]["nIndPointsPerLatent"][1:-1].split(",")]
    nIndPointsPerLatent = nIndPointsPerLatent[:nLatents]
    nTestPoints = int(estInitConfig["control_variables"]["nTestPoints"])
    firstIndPointLoc = float(estInitConfig["control_variables"]["firstIndPointLoc"])
    randomEmbedding = estInitConfig["control_variables"]["randomEmbedding"].lower()=="true"
    initCondIndPointsScale = float(estInitConfig["control_variables"]["initCondIndPointsScale"])
    indPointsLocsKMSEpsilon = float(estInitConfig["control_variables"]["indPointsLocsKMSRegEpsilon"])
    nQuad = int(estInitConfig["control_variables"]["nQuad"])

    optimParamsConfig = estInitConfig._sections["optim_params"]
    optimParams = {}
    optimParams["emMaxIter"] = int(optimParamsConfig["emMaxIter".lower()])
    #
    optimParams["eStepEstimate"] = optimParamsConfig["eStepEstimate".lower()]=="True"
    optimParams["eStepMaxIter"] = int(optimParamsConfig["eStepMaxIter".lower()])
    optimParams["eStepTol"] = float(optimParamsConfig["eStepTol".lower()])
    optimParams["eStepLR"] = float(optimParamsConfig["eStepLR".lower()])
    optimParams["eStepLineSearchFn"] = optimParamsConfig["eStepLineSearchFn".lower()]
    optimParams["eStepNIterDisplay"] = int(optimParamsConfig["eStepNIterDisplay".lower()])
    #
    optimParams["mStepEmbeddingEstimate"] = optimParamsConfig["mStepEmbeddingEstimate".lower()]=="True"
    optimParams["mStepEmbeddingMaxIter"] = int(optimParamsConfig["mStepEmbeddingMaxIter".lower()])
    optimParams["mStepEmbeddingTol"] = float(optimParamsConfig["mStepEmbeddingTol".lower()])
    optimParams["mStepEmbeddingLR"] = float(optimParamsConfig["mStepEmbeddingLR".lower()])
    optimParams["mStepEmbeddingLineSearchFn"] = optimParamsConfig["mStepEmbeddingLineSearchFn".lower()]
    optimParams["mStepEmbeddingNIterDisplay"] = int(optimParamsConfig["mStepEmbeddingNIterDisplay".lower()])
    #
    optimParams["mStepKernelsEstimate"] = optimParamsConfig["mStepKernelsEstimate".lower()]=="True"
    optimParams["mStepKernelsMaxIter"] = int(optimParamsConfig["mStepKernelsMaxIter".lower()])
    optimParams["mStepKernelsTol"] = float(optimParamsConfig["mStepKernelsTol".lower()])
    optimParams["mStepKernelsLR"] = float(optimParamsConfig["mStepKernelsLR".lower()])
    optimParams["mStepKernelsLineSearchFn"] = optimParamsConfig["mStepKernelsLineSearchFn".lower()]
    optimParams["mStepKernelsNIterDisplay"] = int(optimParamsConfig["mStepKernelsNIterDisplay".lower()])
    #
    optimParams["mStepIndPointsEstimate"] = optimParamsConfig["mStepIndPointsEstimate".lower()]=="True"
    optimParams["mStepIndPointsMaxIter"] = int(optimParamsConfig["mStepIndPointsMaxIter".lower()])
    optimParams["mStepIndPointsTol"] = float(optimParamsConfig["mStepIndPointsTol".lower()])
    optimParams["mStepIndPointsLR"] = float(optimParamsConfig["mStepIndPointsLR".lower()])
    optimParams["mStepIndPointsLineSearchFn"] = optimParamsConfig["mStepIndPointsLineSearchFn".lower()]
    optimParams["mStepIndPointsNIterDisplay"] = int(optimParamsConfig["mStepIndPointsNIterDisplay".lower()])
    #
    optimParams["verbose"] = optimParamsConfig["verbose"]=="True"

    if randomEmbedding:
        C0 = torch.rand(nNeurons, nLatents, dtype=torch.double)-0.5*2
        d0 = torch.rand(nNeurons, 1, dtype=torch.double)-0.5*2
    else:
        CFilename = simInitConfig["embedding_params"]["C_filename"]
        dFilename = simInitConfig["embedding_params"]["d_filename"]
        C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, CFilename=CFilename, dFilename=dFilename)
        C0 = C + torch.randn(C.shape)*initCondEmbeddingSTD
        C0 = C0[:,:nLatents]
        d0 = d + torch.randn(d.shape)*initCondEmbeddingSTD

    legQuadPoints, legQuadWeights = utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(nQuad=nQuad, trialsLengths=trialsLengths)

    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=estInitConfig)
    kernelsParams0 = utils.svGPFA.initUtils.getKernelsParams0(kernels=kernels, noiseSTD=0.0)

    qMu0, qSVec0, qSDiag0 = utils.svGPFA.initUtils.getSVPosteriorOnIndPointsParams0(nIndPointsPerLatent=nIndPointsPerLatent, nLatents=nLatents, nTrials=nTrials, scale=initCondIndPointsScale)

    Z0 = utils.svGPFA.initUtils.getIndPointLocs0(nIndPointsPerLatent=nIndPointsPerLatent, trialsLengths=trialsLengths, firstIndPointLoc=firstIndPointLoc)
    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": d0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estimResMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(estResNumber)
        if not os.path.exists(estimResMetaDataFilename):
           estPrefixUsed = False
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)

    kernelsTypes = [type(kernels[k]).__name__ for k in range(len(kernels))]
    estimationDataForMatlabFilename = "results/{:08d}_estimationDataForMatlab.mat".format(estResNumber)
    utils.svGPFA.miscUtils.saveDataForMatlabEstimations(
        qMu0=qMu0, qSVec0=qSVec0, qSDiag0=qSDiag0,
        C0=C0, d0=d0,
        indPointsLocs0=Z0,
        legQuadPoints=legQuadPoints,
        legQuadWeights=legQuadWeights,
        kernelsTypes=kernelsTypes,
        kernelsParams0=kernelsParams0,
        spikesTimes=spikesTimes,
        indPointsLocsKMSEpsilon=indPointsLocsKMSEpsilon,
        trialsLengths=np.array(trialsLengths).reshape(-1,1),
        emMaxIter=optimParams["emMaxIter"],
        eStepMaxIter=optimParams["eStepMaxIter"],
        mStepEmbeddingMaxIter=optimParams["mStepEmbeddingMaxIter"],
        mStepKernelsMaxIter=optimParams["mStepKernelsMaxIter"],
        mStepIndPointsMaxIter=optimParams["mStepIndPointsMaxIter"],
        saveFilename=estimationDataForMatlabFilename)

    # create model
    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels)

    # maximize lower bound
    svEM = stats.svGPFA.svEM.SVEM()
    lowerBoundHist, elapsedTimeHist  = svEM.maximize(
        model=model, measurements=spikesTimes, initialParams=initialParams,
        quadParams=quadParams, optimParams=optimParams,
        indPointsLocsKMSEpsilon=indPointsLocsKMSEpsilon)

    # save estimated values
    estimResConfig = configparser.ConfigParser()
    estimResConfig["simulation_params"] = {"simResNumber": simResNumber}
    estimResConfig["optim_params"] = optimParams
    estimResConfig["estimation_params"] = {"estInitNumber": estInitNumber, "nIndPointsPerLatent": nIndPointsPerLatent}
    with open(estimResMetaDataFilename, "w") as f: estimResConfig.write(f)

    resultsToSave = {"lowerBoundHist": lowerBoundHist, "elapsedTimeHist": elapsedTimeHist, "model": model}
    with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
