import sys
import os
import pdb
import math
import random
import torch
import pickle
import argparse
import configparser

import gcnu_common.utils.config_dict
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.configUtils
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("est_init_number", help="estimation init number",
                        type=int)
    parser.add_argument("--sim_res_number", help="simuluation result number",
                        type=int, default=32451751)
    args = parser.parse_args()
    sim_res_number = args.sim_res_number
    est_init_number = args.est_init_number

    # load data
    sim_res_config_filename = \
        "../results/{:08d}_simulation_metaData.ini".format(sim_res_number)
    sim_res_config = configparser.ConfigParser()
    sim_res_config.read(sim_res_config_filename)

    sim_res_filename = sim_res_config["simulation_results"]["simResFilename"]
    with open(sim_res_filename, "rb") as f:
        simRes = pickle.load(f)
    spikes_times = simRes["spikes"]
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])

    max_spike_time = -math.inf
    min_spike_time = +math.inf
    for r in range(n_trials):
        for n in range(n_neurons):
            spikes_tensor = torch.tensor(spikes_times[r][n])

            if len(spikes_tensor) > 0:
                kr_max_spike_time = spikes_tensor.max()
                if max_spike_time < kr_max_spike_time :
                    max_spike_time = kr_max_spike_time

                kr_min_spike_time = spikes_tensor.min()
                if min_spike_time > kr_min_spike_time :
                    min_spike_time = kr_min_spike_time

    # get initial parameters
    est_init_config_filename = \
        "../data/{:08d}_estimation_metaData.ini".format(est_init_number)
    est_init_config = configparser.ConfigParser()
    est_init_config.read(est_init_config_filename)
    n_latents = int(est_init_config["model_structure_params"]["n_latents"])

    args_info = svGPFA.utils.initUtils.getArgsInfo()
    dynamic_params = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=args, args_info=args_info)

    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=est_init_config).get_dict()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)

    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_latents=n_latents)

    initial_params, quad_params, kernels_types, optim_params = \
        svGPFA.utils.initUtils.getParams(
            n_trials=n_trials, n_neurons=n_neurons,
            dynamic_params=dynamic_params,
            config_file_params=config_file_params,
            default_params=default_params)
    kernels_params0 = initial_params["svPosteriorOnLatents"]["kernelsMatricesStore"]["kernelsParams0"]
    optim_method = optim_params["optim_method"]
    prior_cov_reg_param = optim_params["prior_cov_reg_param"]

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
        qsCholVecs=initial_params["svPosteriorOnLatents"]["svPosteriorOnIndPoints"]["cholVecs"])
    qMu0 = initial_params["svPosteriorOnLatents"]["svPosteriorOnIndPoints"]["mean"]
    qSVec0 = qSVec0
    qSDiag0 = qSDiag0
    C0 = initial_params["svEmbedding"]["C0"]
    d0 = initial_params["svEmbedding"]["d0"]
    Z0 = initial_params["svPosteriorOnLatents"]["kernelsMatricesStore"]["inducingPointsLocs0"]
    legQuadPoints = quad_params["legQuadPoints"]
    legQuadWeights = quad_params["legQuadWeights"]
    trials_start_times, trials_end_times = svGPFA.utils.initUtils.getTrialsStartEndTimes(
        n_trials=n_trials, dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    trials_lengths = [trials_end_times[r] - trials_start_times[r]
                      for r in range(n_trials)]
    if "latentsTrialsTimes" in simRes.keys():
        latentsTrialsTimes = simRes["latentsTrialsTimes"]
    elif "times" in simRes.keys():
        latentsTrialsTimes = simRes["times"]
    else:
        raise ValueError("latentsTrialsTimes or times cannot be found in "
                         f"{sim_res_filename}")

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
        emMaxIter=optim_params["em_max_iter"],
        eStepMaxIter=optim_params["estep_optim_params"]["max_iter"],
        mStepEmbeddingMaxIter=optim_params["mstep_embedding_optim_params"]["max_iter"],
        mStepKernelsMaxIter=optim_params["mstep_kernels_optim_params"]["max_iter"],
        mStepIndPointsMaxIter=optim_params["mstep_indpointslocs_optim_params"]["max_iter"],
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
        svEM.maximize(model=model, optim_params=optim_params, method=optim_method,
                      getIterationModelParamsFn=getKernelParams)

    # save estimated values
    estimResConfig = configparser.ConfigParser()
    estimResConfig["simulation_params"] = {"sim_res_number": sim_res_number}
    estimResConfig["optim_params"] = optim_params
    estimResConfig["estimation_params"] = {
        "est_init_number": est_init_number,
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
