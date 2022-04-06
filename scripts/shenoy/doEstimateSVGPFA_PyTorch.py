import sys
import os
import pdb
import random
import torch
import pickle
import argparse
import configparser
import scipy.io
import numpy as np
import pandas as pd

import shenoyUtils
sys.path.append("../../src")
import stats.svGPFA.svGPFAModelFactory
import stats.svGPFA.svEM
import utils.neuralDataAnalysis
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import utils.svGPFA.initUtils

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("estInitNumber", help="estimation init number", type=int)
    parser.add_argument("--savePartial", help="save partial model estimates",
                        action="store_true")
    parser.add_argument("--location", help="location to analyze", type=int,
                        default=0)
    parser.add_argument("--trials_indices", help="trials indices to analyze",
                        default="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]")
    parser.add_argument("--from_time", help="starting spike analysis time",
                        type=float, default=750.0)
    parser.add_argument("--to_time", help="ending spike analysis time",
                        type=float, default=2500.0)
    parser.add_argument("--min_nSpikes_perNeuron_perTrial",
                        help="min number of spikes per neuron per trial",
                        type=int, default=1)
    parser.add_argument("--save_partial_filename_pattern_pattern",
                        help="pattern for save partial model filename pattern",
                        default="results/{:08d}_{{:s}}_estimatedModel.pickle")
    parser.add_argument("--data_filename", help="data filename",
                        default="~/dev/research/gatsby-swc/datasets/george20040123_hnlds.mat")
    args = parser.parse_args()

    estInitNumber = args.estInitNumber
    save_partial = args.savePartial
    location = args.location
    trials_indices = [int(str) for str in args.trials_indices[1:-1].split(",")]
    from_time = args.from_time
    to_time = args.to_time
    min_nSpikes_perNeuron_perTrial = args.min_nSpikes_perNeuron_perTrial
    save_partial_filename_pattern_pattern = args.save_partial_filename_pattern_pattern
    data_filename = args.data_filename

    spikes_times = shenoyUtils.getSpikesTimes(
        data_filename=data_filename, trials_indices=trials_indices,
        location=location, from_time=from_time, to_time=to_time)
    spikes_times, neurons_indices = utils.neuralDataAnalysis.removeUnitsWithLessSpikesThanThrInAnyTrials(
                spikes_times=spikes_times,
                min_nSpikes_perNeuron_perTrial=
                 min_nSpikes_perNeuron_perTrial)

    nNeurons = len(spikesTimes[0])

    estInitConfigFilename = "data/{:08d}_estimation_metaData.ini".format(estInitNumber)
    estInitConfig = configparser.ConfigParser()
    estInitConfig.read(estInitConfigFilename)
    nLatents = int(estInitConfig["control_variables"]["nLatents"])
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
    optimParams = utils.svGPFA.miscUtils.getOptimParams(optimParamsDict=optimParamsConfig)

    # load data and initial values
    # simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    # simResConfig = configparser.ConfigParser()
    # simResConfig.read(simResConfigFilename)
    # simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]
    # simResFilename = simResConfig["simulation_results"]["simResFilename"]

    # simInitConfig = configparser.ConfigParser()
    # simInitConfig.read(simInitConfigFilename)
    nTrials = len(trials_indices)
    trials_start_times = [from_time for i in range(nTrials)]
    trials_end_times = [to_time for i in range(nTrials)]
    trials_lengths = [trials_end_times[i]-trials_start_times[i] for i in range(nTrials)]

    # with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    # spikesTimes = simRes["spikes"]

    randomEmbedding = estInitConfig["control_variables"]["randomEmbedding"].lower()=="true"
    if randomEmbedding:
        C0 = torch.rand(nNeurons, nLatents, dtype=torch.double).contiguous()
        d0 = torch.rand(nNeurons, 1, dtype=torch.double).contiguous()
    else:
        CFilename = estInitConfig["embedding_params"]["C_filename"]
        dFilename = estInitConfig["embedding_params"]["d_filename"]
        df = pd.read_csv(CFilename, header=None)
        C = df.values
        C = np.delete(C, units_to_remove, axis=0)
        C = torch.from_numpy(C)
        df = pd.read_csv(dFilename, header=None)
        d = df.values
        d = np.delete(d, units_to_remove, axis=0)
        d = torch.from_numpy(d)
        initCondEmbeddingSTD = float(estInitConfig["control_variables"]["initCondEmbeddingSTD"])
        C0 = (C + torch.randn(C.shape)*initCondEmbeddingSTD).contiguous()
        d0 = (d + torch.randn(d.shape)*initCondEmbeddingSTD).contiguous()

    legQuadPoints, legQuadWeights = \
            utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(
                nQuad=nQuad, trials_start_times=trials_start_times,
                trials_end_times=trials_end_times)

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
    save_partial_filename_pattern = save_partial_filename_pattern_pattern.format(estResNumber)

    kernelsTypes = [type(kernels[k]).__name__ for k in range(len(kernels))]
    estimationDataForMatlabFilename = "results/{:08d}_estimationDataForMatlab.mat".format(estResNumber)

    dt_latents = 1.00
    oneSetLatentsTrialTimes = torch.arange(from_time, to_time, dt_latents)
    latentsTrialsTimes = [oneSetLatentsTrialTimes for k in range(nLatents)]
#     if "latentsTrialsTimes" in simRes.keys():
#         latentsTrialsTimes = simRes["latentsTrialsTimes"]
#     elif "times" in simRes.keys():
#         latentsTrialsTimes = simRes["times"]
#     else:
#         raise ValueError("latentsTrialsTimes or times cannot be found in {:s}".format(simResFilename))
    utils.svGPFA.miscUtils.saveDataForMatlabEstimations(
        qMu=qMu0, qSVec=qSVec0, qSDiag=qSDiag0,
        C=C0, d=d0,
        indPointsLocs=Z0,
        legQuadPoints=legQuadPoints,
        legQuadWeights=legQuadWeights,
        kernelsTypes=kernelsTypes,
        kernelsParams=kernelsScaledParams0,
        spikesTimes=spikesTimes,
        indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon,
        trialsLengths=torch.tensor(trials_lengths).reshape(-1,1),
        latentsTrialsTimes=latentsTrialsTimes,
        neurons_indices=neurons_indices,
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
    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModelPyTorch(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels, kernelMatrixInvMethod=kernelMatrixInvMethod,
        indPointsCovRep=indPointsCovRep)

    model.setInitialParamsAndData(measurements=spikesTimes,
                                  initialParams=initialParams,
                                  eLLCalculationParams=quadParams,
                                  indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)

    # save estimated values
    estimResConfig = configparser.ConfigParser()
    # estimResConfig["simulation_params"] = {"simResNumber": simResNumber}
    estimResConfig["data_params"] = {"data_filename": data_filename,
                                     "location": location,
                                     "trials_indices": trials_indices,
                                     "nLatents": nLatents,
                                     "from_time": from_time,
                                     "to_time": to_time}
    estimResConfig["optim_params"] = optimParams
    estimResConfig["estimation_params"] = {"estInitNumber": estInitNumber, "nIndPointsPerLatent": nIndPointsPerLatent}
    with open(estimResMetaDataFilename, "w") as f: estimResConfig.write(f)

    # maximize lower bound
    svEM = stats.svGPFA.svEM.SVEM_PyTorch()
    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
            svEM.maximize(model=model, optimParams=optimParams,
                          method=optimMethod,
                          getIterationModelParamsFn=getKernelParams,
                          savePartial=save_partial, 
                          savePartialFilenamePattern=save_partial_filename_pattern)

    resultsToSave = {"neurons_indices": neurons_indices, "lowerBoundHist": lowerBoundHist, "elapsedTimeHist": elapsedTimeHist, "terminationInfo": terminationInfo, "iterationModelParams": iterationsModelParams, "model": model}
    with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)
    print("Saved results to {:s}".format(modelSaveFilename))

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
