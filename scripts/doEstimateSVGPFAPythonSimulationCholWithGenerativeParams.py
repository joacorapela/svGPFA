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
    parser.add_argument("--savePartial", action="store_true", help="save partial model estimates")
    parser.add_argument("--simulationMetaDataFilenamePattern", default="results/{:08d}_simulation_metaData.ini", help="simulation meta data filename pattern")
    parser.add_argument("--estimationMetaDataFilenamePattern", default="data/{:08d}_estimation_metaData.ini", help="estimation meta data filename pattern")
    parser.add_argument("--estimatedModelFilenamePattern", default="results/{:08d}_estimatedModel.pickle", help="estimated model filename pattern")
    parser.add_argument("--estimatedModelMetaDataFilenamePattern", default="results/{:08d}_estimatedModelMetaData.ini", help="estimated model meta data filename pattern")
    parser.add_argument("--estimatedPartialModelFilenamePattern", default="results/{:08d}_{{:s}}_estimatedModel.pickle", help="estimated partial model filename pattern")
    parser.add_argument("--estimationDataForMatlabFilenamePattern", default="results/{:08d}_estimationDataForMatlab.mat", help="estimation data for Matlab filename pattern")
    args = parser.parse_args()

    simResNumber = args.simResNumber
    estInitNumber = args.estInitNumber
    savePartial = args.savePartial
    simulationMetaDataFilenamePattern = args.simulationMetaDataFilenamePattern
    estimationMetaDataFilenamePattern = args.estimationMetaDataFilenamePattern
    estimatedModelMetaDataFilenamePattern = args.estimatedModelMetaDataFilenamePattern
    estimatedModelFilenamePattern = args.estimatedModelFilenamePattern
    estimatedPartialModelFilenamePattern = args.estimatedPartialModelFilenamePattern
    estimationDataForMatlabFilenamePattern = args.estimationDataForMatlabFilenamePattern

    # load data and initial values
    simResConfigFilename = simulationMetaDataFilenamePattern.format(simResNumber)
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
    indPointsLocsKMSRegEpsilon = float(simInitConfig["control_variables"]["indPointsLocsKMSRegEpsilon"])

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    KzzChol = simRes["KzzChol"]
    indPointsMeans = simRes["indPointsMeans"]
    # C0, d0 = utils.svGPFA.configUtils.getLinearEmbeddingParams(CFilename=simInitConfig["embedding_params"]["C_filename"], dFilename=simInitConfig["embedding_params"]["d_filename"])
    C0, d0 = torch.tensor([[0.5]], dtype=torch.double), torch.tensor([[0]], dtype=torch.double)

    estimationMetaDataFilename = estimationMetaDataFilenamePattern.format(estInitNumber)
    estMetaDataConfig = configparser.ConfigParser()
    estMetaDataConfig.read(estimationMetaDataFilename)
    optimParamsDict = estMetaDataConfig._sections["optim_params"]
    optimParams = utils.svGPFA.miscUtils.getOptimParams(optimParamsDict=optimParamsDict)
    nQuad = int(estMetaDataConfig["control_variables"]["nQuad"])

    legQuadPoints, legQuadWeights = utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(nQuad=nQuad, trialsLengths=trialsLengths)

    # kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=True)
    res = utils.svGPFA.configUtils.getScaledKernels(nLatents=nLatents, config=estMetaDataConfig, forceUnitScale=True)
    kernels = res["kernels"]
    kernelsParamsScales = res["kernelsParamsScales"]
    unscaledKernelsParams0 = utils.svGPFA.initUtils.getKernelsParams0(kernels=kernels, noiseSTD=0.0)

    kernelsParams0 = []
    for i in range(len(unscaledKernelsParams0)):
        kernelsParams0.append(unscaledKernelsParams0[i]/kernelsParamsScales[i])

    Z0 = utils.svGPFA.configUtils.getIndPointsLocs0(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
    # Z0 = utils.svGPFA.configUtils.getIndPointsLocs0(nLatents=nLatents, nTrials=nTrials, config=estMetaDataConfig)
    nIndPointsPerLatent = [Z0[k].shape[1] for k in range(nLatents)]

    # patch to acommodate Lea's equal number of inducing points across trials
    qMu0 = [[] for k in range(nLatents)]
    for k in range(nLatents):
        qMu0[k] = torch.empty((nTrials, nIndPointsPerLatent[k], 1), dtype=torch.double)
        for r in range(nTrials):
            qMu0[k][r,:,:] = indPointsMeans[r][k]
    # end patch

    # diagsSRQSigma0 = utils.svGPFA.configUtils.getDiagsSRQSigma0(config=estMetaDataConfig)
    # srQSigma0s = []
    # for k in range(nLatents):
    #     srQSigma0sForLatent = torch.empty((nTrials, nIndPointsPerLatent[k], nIndPointsPerLatent[k]))
    #     for r in range(nTrials):
    #         srQSigma0sForLatent[r,:,:] = diagsSRQSigma0[k]*torch.eye(nIndPointsPerLatent[k])
    #     srQSigma0s.append(srQSigma0sForLatent)
    # srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVecsFromSRMatrices(srMatrices=srQSigma0s)

    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVecsFromSRMatrices(srMatrices=KzzChol)

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

    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estimResMetaDataFilename = estimatedModelMetaDataFilenamePattern.format(estResNumber)
        if not os.path.exists(estimResMetaDataFilename):
           estPrefixUsed = False
    modelSaveFilename = estimatedModelFilenamePattern.format(estResNumber)

    kernelsTypes = [type(kernels[k]).__name__ for k in range(len(kernels))]
    qSVec0, qSDiag0 = utils.svGPFA.miscUtils.getQSVecsAndQSDiagsFromQSRSigmaVecs(srQSigmaVecs=srQSigma0Vecs)
    estimationDataForMatlabFilename = estimationDataForMatlabFilenamePattern.format(estResNumber)

    utils.svGPFA.miscUtils.saveDataForMatlabEstimations(
        qMu0=qMu0, qSVec0=qSVec0, qSDiag0=qSDiag0,
        C0=C0, d0=d0,
        indPointsLocs0=Z0,
        legQuadPoints=legQuadPoints,
        legQuadWeights=legQuadWeights,
        kernelsTypes=kernelsTypes,
        kernelsParams0=kernelsParams0,
        spikesTimes=spikesTimes,
        indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon,
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
    savePartialFilenamePattern = estimatedPartialModelFilenamePattern.format(estResNumber)
    svEM = stats.svGPFA.svEM.SVEM()
    lowerBoundHist, elapsedTimeHist  = svEM.maximize(
        model=model, measurements=spikesTimes, initialParams=initialParams,
        quadParams=quadParams, optimParams=optimParams,
        indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon,
        savePartial=savePartial, savePartialFilenamePattern=savePartialFilenamePattern)

    # save estimated values
    estimResConfig = configparser.ConfigParser()
    estimResConfig["simulation_params"] = {"simResNumber": simResNumber}
    estimResConfig["estimation_params"] = {"estimationMetaDataFilename": estimationMetaDataFilename}
    with open(estimResMetaDataFilename, "w") as f: estimResConfig.write(f)

    resultsToSave = {"lowerBoundHist": lowerBoundHist, "elapsedTimeHist": elapsedTimeHist, "model": model}
    with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
