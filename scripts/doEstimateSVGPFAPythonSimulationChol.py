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
    scaleForIdentityQSigma0 = float(estInitConfig["control_variables"]["scaleForIdentityQSigma0"])
    indPointsLocsKMSRegEpsilon = float(estInitConfig["control_variables"]["indPointsLocsKMSRegEpsilon"])

    optimParamsConfig = estInitConfig._sections["optim_params"]
    optimMethod = optimParamsConfig["em_method"]
    optimParams = {}
    optimParams["em_max_iter"] = int(optimParamsConfig["em_max_iter"])
    steps = ["estep", "mstep_embedding", "mstep_kernels", "mstep_indpointslocs"]
    for step in steps:
        optimParams["{:s}_estimate".format(step)] = optimParamsConfig["{:s}_estimate".format(step)]=="True"
        optimParams["{:s}_optim_params".format(step)] = {
            "max_iter": int(optimParamsConfig["{:s}_max_iter".format(step)]),
            "lr": float(optimParamsConfig["{:s}_lr".format(step)]),
            "tolerance_grad": float(optimParamsConfig["{:s}_tolerance_grad".format(step)]),
            "tolerance_change": float(optimParamsConfig["{:s}_tolerance_change".format(step)]),
            "line_search_fn": optimParamsConfig["{:s}_line_search_fn".format(step)],
        }
    optimParams["verbose"] = optimParamsConfig["verbose"]=="True"

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

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]

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

    kernels = utils.svGPFA.configUtils.getScaledKernels(nLatents=nLatents, config=estInitConfig, forceUnitScale=True)["kernels"]
    kernelsParams0 = utils.svGPFA.initUtils.getKernelsParams0(kernels=kernels, noiseSTD=0.0)
    kernelsScaledParams0 = utils.svGPFA.initUtils.getKernelsScaledParams0(kernels=kernels, noiseSTD=0.0)
    Z0 = utils.svGPFA.configUtils.getIndPointsLocs0(nLatents=nLatents, nTrials=nTrials, config=estInitConfig)
    nIndPointsPerLatent = [Z0[k].shape[1] for k in range(nLatents)]

    indPointsMeans = utils.svGPFA.configUtils.getVariationalMean0(nLatents=nLatents, nTrials=nTrials, config=estInitConfig)
    # patch to acommodate Lea's equal number of inducing points across trials
    qMu0 = [[] for k in range(nLatents)]
    for k in range(nLatents):
        qMu0[k] = torch.empty((nTrials, nIndPointsPerLatent[k], 1), dtype=torch.double)
        for r in range(nTrials):
            qMu0[k][r,:,:] = indPointsMeans[r][k]
    # end patch

    qSigma0 = utils.svGPFA.configUtils.getVariationalCov0(nLatents=nLatents, nTrials=nTrials, config=estInitConfig)
    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVecsFromSRMatrices(srMatrices=qSigma0)

    qUParams0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
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
    qSVec0, qSDiag0 = utils.svGPFA.miscUtils.getQSVecsAndQSDiagsFromQSRSigmaVecs(srQSigmaVecs=srQSigma0Vecs)
    estimationDataForMatlabFilename = "results/{:08d}_estimationDataForMatlab.mat".format(estResNumber)
    utils.svGPFA.miscUtils.saveDataForMatlabEstimations(
        qMu0=qMu0, qSVec0=qSVec0, qSDiag0=qSDiag0,
        C0=C0, d0=d0,
        indPointsLocs0=Z0,
        legQuadPoints=legQuadPoints,
        legQuadWeights=legQuadWeights,
        kernelsTypes=kernelsTypes,
        kernelsParams0=kernelsScaledParams0,
        spikesTimes=spikesTimes,
        indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon,
        trialsLengths=np.array(trialsLengths).reshape(-1,1),
        emMaxIter=optimParams["em_max_iter"],
        eStepMaxIter=optimParams["estep_optim_params"]["max_iter"],
        mStepEmbeddingMaxIter=optimParams["mstep_embedding_optim_params"]["max_iter"],
        mStepKernelsMaxIter=optimParams["mstep_kernels_optim_params"]["max_iter"],
        mStepIndPointsMaxIter=optimParams["mstep_indpointslocs_optim_params"]["max_iter"],
        saveFilename=estimationDataForMatlabFilename)

    def getKernelParams(model):
        kernelParams = model.getKernelsParams()[0]
        return kernelParams

    def embeddingParamsLogPrior(embeddingParams):
        def cElementLogPrior(x, min=0.1, max=10.0):
            uniform = torch.distributions.uniform.Uniform(min, max)
            answer = uniform.log_prob(x)
            return answer

        def dElementLogPrior(x, mean=0.0, sd=10.0):
            normal = torch.distributions.normal.Normal(loc=mean, scale=sd)
            answer = normal.log_prob(x)
            return answer

        C = embeddingParams[0]
        ClogProb = 0.0
        for row in C:
            for value in row:
                ClogProb = ClogProb + cElementLogPrior(x=value)

        d = embeddingParams[1]
        dLogProb = 0.0
        for value in d:
                dLogProb = dLogProb + dElementLogPrior(x=value)

        answer = ClogProb + dLogProb
        return answer

    def kernelsParamsLogPrior(kernelsParams):
        def periodLogPrior(x, mean=5.0, sd=0.25):
            normal = torch.distributions.normal.Normal(loc=mean, scale=sd)
            answer = normal.log_prob(x)
            return answer

        def lengthscaleLogPrior(x, mean=2.25, sd=1.0):
            normal = torch.distributions.normal.Normal(loc=mean, scale=sd)
            answer = normal.log_prob(x)
            return answer

        lengthscale = kernelsParams[0][0]
        lengthscaleLogProb = lengthscaleLogPrior(lengthscale)

        period = kernelsParams[0][1]
        periodLogProb = periodLogPrior(period)

        answer = lengthscaleLogProb + periodLogProb
        return answer

    def indPointsLocsLogPrior(indPointsLocs):
        def elementLogPrior(x):
            uniform = torch.distributions.uniform.Uniform(0, torch.tensor(trialsLengths).max())
            answer = uniform.log_prob(x)
            return answer

        indPointsLocsLogProb = 0.0
        for value in indPointsLocs[0][0,:,0]:
            indPointsLocsLogProb = indPointsLocsLogProb + elementLogPrior(value)
        return indPointsLocsLogProb

    paramsLogPriors = {
        "embedding": embeddingParamsLogPrior,
        "kernels": kernelsParamsLogPrior,
        "indPointsLocs": indPointsLocsLogPrior,
    }

    # create model
    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels,
        paramsLogPriors=paramsLogPriors)

    model.setInitialParamsAndData(measurements=spikesTimes,
                                  initialParams=initialParams,
                                  quadParams=quadParams,
                                  indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)

    # maximize lower bound
    svEM = stats.svGPFA.svEM.SVEM()
    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = svEM.maximize(model=model, optimParams=optimParams, method=optimMethod, getIterationModelParamsFn=getKernelParams)

    # save estimated values
    estimResConfig = configparser.ConfigParser()
    estimResConfig["simulation_params"] = {"simResNumber": simResNumber}
    estimResConfig["optim_params"] = optimParams
    estimResConfig["estimation_params"] = {"estInitNumber": estInitNumber, "nIndPointsPerLatent": nIndPointsPerLatent}
    with open(estimResMetaDataFilename, "w") as f: estimResConfig.write(f)

    resultsToSave = {"lowerBoundHist": lowerBoundHist, "elapsedTimeHist": elapsedTimeHist, "terminationInfo": terminationInfo, "iterationModelParams": iterationsModelParams, "model": model}
    with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)
    print("Saved results to {:s}".format(modelSaveFilename))

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
