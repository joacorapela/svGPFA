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
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import utils.svGPFA.initUtils

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number",
                        type=int)
    parser.add_argument("--intermediateDesc",
                        help="Descriptor of the intermediate model estimate to plot",
                        type=str, default="None")
    parser.add_argument("--finalEstimatedModelFilenamePattern",
                        help="final estimated model filename pattern",
                        default="results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--intermediateEstimatedModelFilenamePattern",
                        help="intermediate estimated model filename pattern",
                        default="results/{:08d}_{:s}_estimatedModel.pickle")
    parser.add_argument("--estimatedModelMetadataFilenamePattern",
                        help="metadata for estimated model filename pattern",
                        default="results/{:08d}_estimation_metaData.ini")
    parser.add_argument("--estInitFilenamePattern",
                        help="estimation initialization filename pattern",
                        default="data/{:08d}_estimation_metaData.ini")
    parser.add_argument("--final_estDataForMatlab_filenamePattern",
                        help="final estimation data for Matlab filename pattern",
                        default="results/{:08d}_final_estimationDataForMatlab.mat")
    parser.add_argument("--intermediate_estDataForMatlab_filenamePattern",
                        help="intermediate estimation data for Matlab filename pattern",
                        default="results/{:08d}_{:s}_estimationDataForMatlab.mat")
    args = parser.parse_args()

    estResNumber = args.estResNumber
    intermediateDesc = args.intermediateDesc
    finalEstimatedModelFilenamePattern = args.finalEstimatedModelFilenamePattern
    intermediateEstimatedModelFilenamePattern = \
            args.intermediateEstimatedModelFilenamePattern
    estimatedModelMetadataFilenamePattern = \
            args.estimatedModelMetadataFilenamePattern
    estInitFilenamePattern = args.estInitFilenamePattern
    finalEstDataForMatlab_filenamePattern = args.final_estDataForMatlab_filenamePattern
    intermediateEstDataForMatlab_filenamePattern = args.intermediate_estDataForMatlab_filenamePattern
    estMetaDataFilename = estimatedModelMetadataFilenamePattern.format(estResNumber, intermediateDesc)

    if intermediateDesc=="None":
        estimatedModelFilename = finalEstimatedModelFilenamePattern.format(estResNumber)
        estDataForMatlabFilename = finalEstDataForMatlab_filenamePattern.format(estResNumber)
    else:
        estimatedModelFilename = intermediateEstimatedModelFilenamePattern.format(estResNumber, intermediateDesc)
        estDataForMatlabFilename = intermediateEstDataForMatlab_filenamePattern.format(estResNumber, intermediateDesc)


    estMetaDataConfig = configparser.ConfigParser()
    estMetaDataConfig.read(estMetaDataFilename)
    nLatents = int(estMetaDataConfig["data_params"]["nLatents"])
    data_filename = estMetaDataConfig["data_params"]["data_filename"]
    location = int(estMetaDataConfig["data_params"]["location"])
    trials_str = estMetaDataConfig["data_params"]["trials"]
    trials = [int(str) for str in trials_str[1:-1].split(",")]
    nTrials = len(trials)
    from_time = float(estMetaDataConfig["data_params"]["from_time"])
    to_time = float(estMetaDataConfig["data_params"]["to_time"])

    estInitNumber = int(estMetaDataConfig["estimation_params"]["estInitNumber"])
    estInitFilename = estInitFilenamePattern.format(estInitNumber)
    estInitConfig = configparser.ConfigParser()
    estInitConfig.read(estInitFilename)

    indPointsLocsKMSRegEpsilon = float(estInitConfig["control_variables"]["indPointsLocsKMSRegEpsilon"])
    nQuad = int(estInitConfig["control_variables"]["nQuad"])
    trialsLengths = [to_time-from_time for i in range(nTrials)]

    with open(estimatedModelFilename, "rb") as f: model = pickle.load(f)["model"]

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


    svPosteriorOnIndPointsParams = model.getSVPosteriorOnIndPointsParams()
    qMu = svPosteriorOnIndPointsParams[:nLatents]
    qSVec = svPosteriorOnIndPointsParams[nLatents:(2*nLatents)]
    qSDiag = svPosteriorOnIndPointsParams[(2*nLatents):(3*nLatents)]

    embeddingParams = model.getSVEmbeddingParams()
    C = embeddingParams[0]
    d = embeddingParams[1]
    Z = model.getIndPointsLocs()

    legQuadPoints, legQuadWeights = utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(nQuad=nQuad, trialsLengths=trialsLengths)

    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=estInitConfig, forceUnitScale=True)
    kernelsTypes = [type(kernels[k]).__name__ for k in range(len(kernels))]      

    kernels = model._eLL._svEmbeddingAllTimes._svPosteriorOnLatents._indPointsLocsKMS._kernels
    kernelsScaledParams = utils.svGPFA.initUtils.getKernelsScaledParams0(kernels=kernels, noiseSTD=0.0)

    mat = scipy.io.loadmat(os.path.expanduser(data_filename))
    spikesTimes = shenoyUtils.getTrialsAndLocationSpikesTimes(mat=mat,
                                                               trials=trials,
                                                               location=location)
    spikesTimes = miscUtils.clipSpikesTimes(spikes_times=spikesTimes,
                                             from_time=from_time, to_time=to_time)

    dt_latents = 0.01
    oneSetLatentsTrialTimes = torch.arange(from_time, to_time, dt_latents)
    latentsTrialsTimes = [oneSetLatentsTrialTimes for k in range(nLatents)]
    pdb.set_trace()

    utils.svGPFA.miscUtils.saveDataForMatlabEstimations(
        qMu=qMu, qSVec=qSVec, qSDiag=qSDiag,
        C=C, d=d,
        indPointsLocs=Z,
        legQuadPoints=legQuadPoints,
        legQuadWeights=legQuadWeights,
        kernelsTypes=kernelsTypes,
        kernelsParams=kernelsScaledParams,
        spikesTimes=spikesTimes,
        indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon,
        trialsLengths=torch.tensor(trialsLengths).reshape(-1,1),
        latentsTrialsTimes=latentsTrialsTimes,
        emMaxIter=optimParams["em_max_iter"],
        eStepMaxIter=optimParams["estep_optim_params"]["max_iter"],
        mStepEmbeddingMaxIter=optimParams["mstep_embedding_optim_params"]["max_iter"],
        mStepKernelsMaxIter=optimParams["mstep_kernels_optim_params"]["max_iter"],
        mStepIndPointsMaxIter=optimParams["mstep_indpointslocs_optim_params"]["max_iter"],
        saveFilename=estDataForMatlabFilename)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
