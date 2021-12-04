import sys
import os
import pdb
import random
import torch
import pickle
import argparse
import configparser
import scipy.io

import shenoyUtils
import miscUtils
sys.path.append("../../src")
import stats.svGPFA.svGPFAModelFactory
import stats.svGPFA.svEM
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import utils.svGPFA.initUtils

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("estInitNumber", help="estimation init number", type=int)
    parser.add_argument("--data_filename", help="data filename",
                        default="~/dev/research/gatsby-swc/datasets/george20040123_hnlds.mat")
    parser.add_argument("--location", help="location to analyze", type=int,
                        default=0)
    parser.add_argument("--trials", help="trials to analyze",
                        default="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]")
    parser.add_argument("--nLatents", help="number of latent variables", type=int,
                        default=2)
    parser.add_argument("--from_time", help="starting spike analysis time",
                        type=float, default=0.750)
    parser.add_argument("--to_time", help="ending spike analysis time",
                        type=float, default=2.250)

    args = parser.parse_args()

    estInitNumber = args.estInitNumber
    data_filename = args.data_filename
    location = args.location
    trials = [int(str) for str in args.trials[1:-1].split(",")]
    nLatents = args.nLatents
    from_time = args.from_time
    to_time = args.to_time

    mat = scipy.io.loadmat(os.path.expanduser(data_filename))
    spikesTimes = shenoyUtils.getTrialsAndLocationSpikesTimes(mat=mat,
                                                               trials=trials,
                                                               location=location)
    spikesTimes = miscUtils.clipSpikesTimes(spikes_times=spikesTimes,
                                             from_time=from_time, to_time=to_time)

    estInitConfigFilename = "data/{:08d}_estimation_metaData.ini".format(estInitNumber)
    estInitConfig = configparser.ConfigParser()
    estInitConfig.read(estInitConfigFilename)
    nQuad = int(estInitConfig["control_variables"]["nQuad"])
    kernelMatrixInvMethodStr = estInitConfig["control_variables"]["kernelMatrixInvMethod"]
    indPointsCovRepStr = estInitConfig["control_variables"]["indPointsCovRep"]
    if kernelMatrixInvMethodStr == "Chol":
        kernelMatrixInvMethod = stats.svGPFA.svGPFAModelFactory.kernelMatrixInvChol
    elif kernelMatrixInvMethodStr == "PInv":
        kernelMatrixInvMethod = stats.svGPFA.svGPFAModelFactory.kernelMatrixInvPInv
    else:
        raise RuntimeError("Invalid kernelMatrixInvMethod={:s}".format(kernelMatrixInvMethodStr))
    if indPointsCovRepStr == "Chol":
        indPointsCovRep = stats.svGPFA.svGPFAModelFactory.indPointsCovChol
    elif indPointsCovRepStr == "Rank1PlusDiag":
        indPointsCovRep = stats.svGPFA.svGPFAModelFactory.indPointsCovRank1PlusDiag
    else:
        raise RuntimeError("Invalid indPointsCovRep={:s}".format(indPointsCovRepStr))
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
    # simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    # simResConfig = configparser.ConfigParser()
    # simResConfig.read(simResConfigFilename)
    # simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]
    # simResFilename = simResConfig["simulation_results"]["simResFilename"]

    # simInitConfig = configparser.ConfigParser()
    # simInitConfig.read(simInitConfigFilename)
    nNeurons = len(spikesTimes[0])
    nTrials = len(trials)
    trialsLengths = [to_time-from_time for i in range(nTrials)]

    # with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    # spikesTimes = simRes["spikes"]

    randomEmbedding = estInitConfig["control_variables"]["randomEmbedding"].lower()=="true"
    if randomEmbedding:
        C0 = torch.rand(nNeurons, nLatents, dtype=torch.double).contiguous()
        d0 = torch.rand(nNeurons, 1, dtype=torch.double).contiguous()
    else:
        CFilename = estInitConfig["embedding_params"]["C_filename"]
        dFilename = estInitConfig["embedding_params"]["d_filename"]
        C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(CFilename=CFilename, dFilename=dFilename)
        initCondEmbeddingSTD = float(estInitConfig["control_variables"]["initCondEmbeddingSTD"])
        C0 = (C + torch.randn(C.shape)*initCondEmbeddingSTD).contiguous()
        d0 = (d + torch.randn(d.shape)*initCondEmbeddingSTD).contiguous()

    legQuadPoints, legQuadWeights = utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(nQuad=nQuad, trialsLengths=trialsLengths)

    # kernels = utils.svGPFA.configUtils.getScaledKernels(nLatents=nLatents, config=estInitConfig, forceUnitScale=True)["kernels"]
    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=estInitConfig, forceUnitScale=True)
    kernelsScaledParams0 = utils.svGPFA.initUtils.getKernelsScaledParams0(kernels=kernels, noiseSTD=0.0)
    Z0 = utils.svGPFA.configUtils.getIndPointsLocs0(nLatents=nLatents, nTrials=nTrials, config=estInitConfig)
    nIndPointsPerLatent = [Z0[k].shape[1] for k in range(nLatents)]

    qMu0 = utils.svGPFA.configUtils.getVariationalMean0(nLatents=nLatents, nTrials=nTrials, config=estInitConfig)
#     indPointsMeans = utils.svGPFA.configUtils.getVariationalMean0(nLatents=nLatents, nTrials=nTrials, config=estInitConfig)
#     # patch to acommodate Lea's equal number of inducing points across trials
#     qMu0 = [[] for k in range(nLatents)]
#     for k in range(nLatents):
#         qMu0[k] = torch.empty((nTrials, nIndPointsPerLatent[k], 1), dtype=torch.double)
#         for r in range(nTrials):
#             qMu0[k][r,:,:] = indPointsMeans[k][r]
#     # end patch

    qSigma0 = utils.svGPFA.configUtils.getVariationalCov0(nLatents=nLatents, nTrials=nTrials, config=estInitConfig)
    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVecsFromSRMatrices(srMatrices=qSigma0)
    qSVec0, qSDiag0 = utils.svGPFA.miscUtils.getQSVecsAndQSDiagsFromQSRSigmaVecs(srQSigmaVecs=srQSigma0Vecs)

    if indPointsCovRep==stats.svGPFA.svGPFAModelFactory.indPointsCovChol:
        qUParams0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
    elif  indPointsCovRep==stats.svGPFA.svGPFAModelFactory.indPointsCovRank1PlusDiag:
        qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    else:
        raise RuntimeError("Invalid indPointsCovRep")

    kmsParams0 = {"kernelsParams0": kernelsScaledParams0,
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

    dt_latents = 0.01
    oneSetLatentsTrialTimes = torch.arange(from_time, to_time, dt_latents)
    latentsTrialsTimes = [oneSetLatentsTrialTimes for k in range(nLatents)]
#     if "latentsTrialsTimes" in simRes.keys():
#         latentsTrialsTimes = simRes["latentsTrialsTimes"]
#     elif "times" in simRes.keys():
#         latentsTrialsTimes = simRes["times"]
#     else:
#         raise ValueError("latentsTrialsTimes or times cannot be found in {:s}".format(simResFilename))
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
        trialsLengths=torch.tensor(trialsLengths).reshape(-1,1),
        latentsTrialsTimes=latentsTrialsTimes,
        emMaxIter=optimParams["em_max_iter"],
        eStepMaxIter=optimParams["estep_optim_params"]["max_iter"],
        mStepEmbeddingMaxIter=optimParams["mstep_embedding_optim_params"]["max_iter"],
        mStepKernelsMaxIter=optimParams["mstep_kernels_optim_params"]["max_iter"],
        mStepIndPointsMaxIter=optimParams["mstep_indpointslocs_optim_params"]["max_iter"],
        saveFilename=estimationDataForMatlabFilename)

    def getKernelParams(model):
        kernelParams = model.getKernelsParams()[0]
        return kernelParams

    # create model
    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels, kernelMatrixInvMethod=kernelMatrixInvMethod,
        indPointsCovRep=indPointsCovRep)

    model.setInitialParamsAndData(measurements=spikesTimes,
                                  initialParams=initialParams,
                                  eLLCalculationParams=quadParams,
                                  indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)

    # maximize lower bound
    svEM = stats.svGPFA.svEM.SVEM()
    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = svEM.maximize(model=model, optimParams=optimParams, method=optimMethod, getIterationModelParamsFn=getKernelParams)

    # save estimated values
    estimResConfig = configparser.ConfigParser()
    # estimResConfig["simulation_params"] = {"simResNumber": simResNumber}
    estimResConfig["data_params"] = {"data_filename": data_filename,
                                     "location": location,
                                     "trials": trials,
                                     "nLatents": nLatents,
                                     "from_time": from_time,
                                     "to_time": to_time}
    estimResConfig["optim_params"] = optimParams
    estimResConfig["estimation_params"] = {"estInitNumber": estInitNumber, "nIndPointsPerLatent": nIndPointsPerLatent}
    with open(estimResMetaDataFilename, "w") as f: estimResConfig.write(f)

    resultsToSave = {"lowerBoundHist": lowerBoundHist, "elapsedTimeHist": elapsedTimeHist, "terminationInfo": terminationInfo, "iterationModelParams": iterationsModelParams, "model": model}
    with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)
    print("Saved results to {:s}".format(modelSaveFilename))

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
