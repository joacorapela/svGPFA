import sys
import os
import pdb
import random
import torch
import pickle
import argparse
import configparser

import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.configUtils
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("simResNumber", help="simuluation result number",
                        type=int)
    parser.add_argument("estInitNumber", help="estimation init number",
                        type=int)
    args = parser.parse_args()

    simResNumber = args.simResNumber
    estInitNumber = args.estInitNumber

    estInitConfigFilename = \
        "../data/{:08d}_estimation_metaData.ini".format(estInitNumber)
    est_init_config = configparser.ConfigParser()
    est_init_config.read(estInitConfigFilename)

    # load data
    simResConfigFilename = \
        "../results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simInitConfigFilename = \
        simResConfig["simulation_params"]["simInitConfigFilename"]
    simResFilename = simResConfig["simulation_results"]["simResFilename"]
    with open(simResFilename, "rb") as f:
        simRes = pickle.load(f)
    spikes_times = simRes["spikes"]

    # get initial parameters
    prior_cov_reg_param = \
        float(est_init_config["control_variables"]["prior_cov_reg_param"])
    initial_params, quad_params, kernels_types = \
        svGPFA.utils.initUtils.getInitialAndQuadParamsAndKernelsTypes(
            config=est_init_config)
    kernels_params0 = initial_params["svPosteriorOnLatents"]["kernelsMatricesStore"]["kernelsParams0"]

    # get optimization parameters
    optim_params_config = est_init_config._sections["optim_params"]
    optim_method = optim_params_config["em_method"]
    optimParams = svGPFA.utils.configUtils.getOptimParams(
        optim_params_config=optim_params_config)

    # build modelSaveFilename
    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estimResMetaDataFilename = "../results/{:08d}_estimation_metaData.ini".format(estResNumber)
        if not os.path.exists(estimResMetaDataFilename):
            estPrefixUsed = False
    modelSaveFilename = "../results/{:08d}_estimatedModel.pickle".format(estResNumber)

    # save data for Matlab estimation
    qSVec0, qSDiag0 = svGPFA.utils.miscUtils.getQSVecsAndQSDiagsFromQSCholVecs(
        qsCholVecs=initial_params["svPosteriorOnLatents"]["svPosteriorOnIndPoints"]["srQSigma0Vecs"])
    qMu0 = initial_params["svPosteriorOnLatents"]["svPosteriorOnIndPoints"]["qMu0"]
    qSVec0 = qSVec0
    qSDiag0 = qSDiag0
    C0 = initial_params["svEmbedding"]["C0"]
    d0 = initial_params["svEmbedding"]["d0"]
    Z0 = initial_params["svPosteriorOnLatents"]["kernelsMatricesStore"]["inducingPointsLocs0"]
    legQuadPoints = quad_params["legQuadPoints"]
    legQuadWeights = quad_params["legQuadWeights"]
    n_trials = int(est_init_config["control_variables"]["n_trials"])
    trials_start_times, trials_end_times = svGPFA.utils.initUtils.getTrialsStartEndTimes(
        n_trials=n_trials, config=est_init_config)
    trials_lengths = [trials_end_times[r] - trials_start_times[r]
                      for r in range(n_trials)]
    if "latentsTrialsTimes" in simRes.keys():
        latentsTrialsTimes = simRes["latentsTrialsTimes"]
    elif "times" in simRes.keys():
        latentsTrialsTimes = simRes["times"]
    else:
        raise ValueError("latentsTrialsTimes or times cannot be found in {:s}".format(simResFilename))

    estimationDataForMatlabFilename = "../results/{:08d}_estimationDataForMatlab.mat".format(estResNumber)
    svGPFA.utils.miscUtils.saveDataForMatlabEstimations(
        qMu=qMu0, qSVec=qSVec0, qSDiag=qSDiag0,
        C=C0, d=d0,
        indPointsLocs=Z0,
        legQuadPoints=legQuadPoints,
        legQuadWeights=legQuadWeights,
        kernelsTypes=kernels_types,
        kernelsParams=kernels_params0,
        spikesTimes=spikes_times,
        indPointsLocsKMSRegEpsilon=prior_cov_reg_param,
        trialsLengths=torch.tensor(trials_lengths).reshape(-1, 1),
        latentsTrialsTimes=latentsTrialsTimes,
        emMaxIter=optimParams["em_max_iter"],
        eStepMaxIter=optimParams["estep_optim_params"]["max_iter"],
        mStepEmbeddingMaxIter=optimParams["mstep_embedding_optim_params"]["max_iter"],
        mStepKernelsMaxIter=optimParams["mstep_kernels_optim_params"]["max_iter"],
        mStepIndPointsMaxIter=optimParams["mstep_indpointslocs_optim_params"]["max_iter"],
        saveFilename=estimationDataForMatlabFilename)

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(kernels_types=kernels_types,
                                                  kernels_params=kernels_params0)

    # create model
    kernelMatrixInvMethod = svGPFA.stats.svGPFAModelFactory.kernelMatrixInvChol
    indPointsCovRep = svGPFA.stats.svGPFAModelFactory.indPointsCovChol
    model = svGPFA.stats.svGPFAModelFactory.SVGPFAModelFactory.buildModelPyTorch(
        conditionalDist=svGPFA.stats.svGPFAModelFactory.PointProcess,
        linkFunction=svGPFA.stats.svGPFAModelFactory.ExponentialLink,
        embeddingType=svGPFA.stats.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels, kernelMatrixInvMethod=kernelMatrixInvMethod,
        indPointsCovRep=indPointsCovRep)

    model.setInitialParamsAndData(
        measurements=spikes_times,
        initialParams=initial_params,
        eLLCalculationParams=quad_params,
        priorCovRegParam=prior_cov_reg_param)

    # maximize lower bound
    def getKernelParams(model):
        kernelParams = model.getKernelsParams()[0]
        return kernelParams

    svEM = svGPFA.stats.svEM.SVEM_PyTorch()
    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
        svEM.maximize(model=model, optimParams=optimParams, method=optim_method,
                      getIterationModelParamsFn=getKernelParams)

    # save estimated values
    estimResConfig = configparser.ConfigParser()
    estimResConfig["simulation_params"] = {"simResNumber": simResNumber}
    estimResConfig["optim_params"] = optimParams
    estimResConfig["estimation_params"] = {
        "estInitNumber": estInitNumber,
    }
    with open(estimResMetaDataFilename, "w") as f:
        estimResConfig.write(f)

    resultsToSave = {"lowerBoundHist": lowerBoundHist,
                     "elapsedTimeHist": elapsedTimeHist,
                     "terminationInfo": terminationInfo,
                     "iterationModelParams": iterationsModelParams,
                     "model": model}
    with open(modelSaveFilename, "wb") as f:
        pickle.dump(resultsToSave, f)
    print("Saved results to {:s}".format(modelSaveFilename))

    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
