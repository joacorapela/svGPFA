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
    parser.add_argument("simResNumber", help="simuluation result number", type=int)
    parser.add_argument("estInitNumber", help="estimation init number", type=int)
    args = parser.parse_args()

    simResNumber = args.simResNumber
    estInitNumber = args.estInitNumber

    estInitConfigFilename = "data/{:08d}_estimation_metaData.ini".format(estInitNumber)
    estInitConfig = configparser.ConfigParser()
    estInitConfig.read(estInitConfigFilename)
    nQuad = int(estInitConfig["control_variables"]["nQuad"])

    optimParamsConfig = estInitConfig._sections["optim_params"]
    optimParams = {}
    optimParams["em_max_iter"] = int(optimParamsConfig["em_max_iter"])
    steps = ["estep", "mstep_embedding", "mstep_kernels", "mstep_indpointslocs"]
    for step in steps:
        optimParams["{:s}_estimate".format(step)] = optimParamsConfig["{:s}_estimate".format(step)]=="True"
        optimParams["{:s}_max_iter".format(step)] = int(optimParamsConfig["{:s}_max_iter".format(step)])
        optimParams["{:s}_lr".format(step)] = float(optimParamsConfig["{:s}_lr".format(step)])
        optimParams["{:s}_tol".format(step)] = float(optimParamsConfig["{:s}_tol".format(step)])
        optimParams["{:s}_niter_display".format(step)] = int(optimParamsConfig["{:s}_niter_display".format(step)])
        optimParams["{:s}_line_search_fn".format(step)] = optimParamsConfig["{:s}_line_search_fn".format(step)]
    optimParams["verbose"] = optimParamsConfig["verbose"]=="True"

    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estimResMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(estResNumber)
        if not os.path.exists(estimResMetaDataFilename):
           estPrefixUsed = False
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)

    # load data and initial values
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
    # firstIndPointLoc = float(simInitConfig["control_variables"]["firstIndPointLoc"])
    indPointsLocsKMSRegEpsilon = float(simInitConfig["control_variables"]["indPointsLocsKMSRegEpsilon"])

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    KzzChol = simRes["KzzChol"]
    indPointsMeans = simRes["indPointsMeans"]
    randomEmbedding = estInitConfig["control_variables"]["randomEmbedding"].lower()=="true"
    if randomEmbedding:
        C0 = torch.rand(nNeurons, nLatents, dtype=torch.double)-0.5*2
        d0 = torch.rand(nNeurons, 1, dtype=torch.double)-0.5*2
    else:
        CFilename = estInitConfig["embedding_params"]["C_filename"]
        dFilename = estInitConfig["embedding_params"]["d_filename"]
        C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(CFilename=CFilename, dFilename=dFilename)
        initCondEmbeddingSTD = float(estInitConfig["control_variables"]["initCondEmbeddingSTD"])
        C0 = C + torch.randn(C.shape)*initCondEmbeddingSTD
        d0 = d + torch.randn(d.shape)*initCondEmbeddingSTD

    legQuadPoints, legQuadWeights = utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(nQuad=nQuad, trialsLengths=trialsLengths)

    # kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=True)
    # kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=estInitConfig, forceUnitScale=True)
    res = utils.svGPFA.configUtils.getScaledKernels(nLatents=nLatents, config=estInitConfig, forceUnitScale=True)
    kernels = res["kernels"]
    kernelsParamsScales = res["kernelsParamsScales"]
    unscaledKernelsParams0 = utils.svGPFA.initUtils.getKernelsParams0(kernels=kernels, noiseSTD=0.0)

    kernelsParams0 = []
    for i in range(len(unscaledKernelsParams0)):
        kernelsParams0.append(unscaledKernelsParams0[i]/kernelsParamsScales[i])

    # Z0 = utils.svGPFA.configUtils.getIndPointsLocs0(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
    Z0 = utils.svGPFA.configUtils.getIndPointsLocs0(nLatents=nLatents, nTrials=nTrials, config=estInitConfig)
    nIndPointsPerLatent = [Z0[k].shape[1] for k in range(nLatents)]

    # patch to acommodate Lea's equal number of inducing points across trials
    qMu0 = [[] for k in range(nLatents)]
    for k in range(nLatents):
        qMu0[k] = torch.empty((nTrials, nIndPointsPerLatent[k], 1), dtype=torch.double)
        for r in range(nTrials):
            qMu0[k][r,:,:] = indPointsMeans[r][k]
    # end patch

    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVecsFromSRMatrices(srMatrices=KzzChol)
    # epsilonSRQSigma0 = 1e4
    # srQSigma0s = []
    # for k in range(nLatents):
    #     srQSigma0sForLatent = torch.empty((nTrials, nIndPointsPerLatent[k], nIndPointsPerLatent[k]))
    #     for r in range(nTrials):
    #         srQSigma0sForLatent[r,:,:] = epsilonSRQSigma0*torch.eye(nIndPointsPerLatent[k])
    #     srQSigma0s.append(srQSigma0sForLatent)
    # srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVecsFromSRMatrices(srMatrices=srQSigma0s)

    qUParams0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
    kmsParams0 = {"kernelsParams0": unscaledKernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": d0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    kernelsTypes = [type(kernels[k]).__name__ for k in range(len(kernels))]
    qSVec0, qSDiag0 = utils.svGPFA.miscUtils.getQSVecsAndQSDiagsFromQSRSigmaVecs(srQSigmaVecs=srQSigma0Vecs)
    estimationDataForMatlabFilename = "results/{:08d}_estimationDataForMatlab.mat".format(estResNumber)

#     utils.svGPFA.miscUtils.saveDataForMatlabEstimations(
#         qMu0=qMu0, qSVec0=qSVec0, qSDiag0=qSDiag0,
#         C0=C, d0=d,
#         indPointsLocs0=Z0,
#         legQuadPoints=legQuadPoints,
#         legQuadWeights=legQuadWeights,
#         kernelsTypes=kernelsTypes,
#         kernelsParams0=kernelsParams0,
#         spikesTimes=spikesTimes,
#         indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon,
#         trialsLengths=np.array(trialsLengths).reshape(-1,1),
#         emMaxIter=optimParams["emMaxIter"],
#         eStepMaxIter=optimParams["eStepMaxIter"],
#         mStepEmbeddingMaxIter=optimParams["mStepEmbeddingMaxIter"],
#         mStepKernelsMaxIter=optimParams["mStepKernelsMaxIter"],
#         mStepIndPointsMaxIter=optimParams["mStepIndPointsMaxIter"],
#         saveFilename=estimationDataForMatlabFilename)

    # create model
    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels)

    model.setInitialParamsAndData(measurements=spikesTimes,
                                  initialParams=initialParams,
                                  quadParams=quadParams,
                                  indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)

    # maximize lower bound
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)
    savePartialFilenamePattern = "results/{:08d}_{{:s}}_estimatedModel.pickle".format(estResNumber)
    svEM = stats.svGPFA.svEM.SVEM()
    lowerBoundHist, elapsedTimeHist  = svEM.maximize(
        model=model, optimParams=optimParams,
        savePartial=True, savePartialFilenamePattern=savePartialFilenamePattern)

    # save estimated values
    estimResConfig = configparser.ConfigParser()
    estimResConfig["simulation_params"] = {"simResNumber": simResNumber}
    estimResConfig["optim_params"] = optimParams
    estimResConfig["estimation_params"] = {"estInitNumber": estInitNumber, "nIndPointsPerLatent": nIndPointsPerLatent}
    with open(estimResMetaDataFilename, "w") as f: estimResConfig.write(f)

    resultsToSave = {"lowerBoundHist": lowerBoundHist, "elapsedTimeHist": elapsedTimeHist, "model": model}
    # with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
