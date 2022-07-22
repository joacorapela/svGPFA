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
    parser.add_argument("--sim_res_number", help="simuluation result number",
                        type=int, default=32451751)
    parser.add_argument("--sim_res_filename_pattern",
                        help="simuluation result filename pattern",
                        type=str, default="../data/{:08d}_simRes.pickle")
    parser.add_argument("--est_init_number", help="estimation init number",
                        type=int, default=545)
    parser.add_argument("--est_init_config_filename_pattern",
                        help="estimation initialization configuration filename pattern",
                        type=str,
                        default="../params/{:08d}_estimation_metaData.ini")

    args = parser.parse_args()
    sim_res_number = args.sim_res_number
    est_init_number = args.est_init_number
    sim_res_filename_pattern = args.sim_res_filename_pattern
    est_init_config_filename_pattern  = args.est_init_config_filename_pattern

    # load data
    sim_res_filename = sim_res_filename_pattern.format(sim_res_number)
    with open(sim_res_filename, "rb") as f:
        simRes = pickle.load(f)
    spikes_times = simRes["spikes"]
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])

    # get initial parameters
    est_init_config_filename = est_init_config_filename_pattern.format(
        est_init_number)
    est_init_config = configparser.ConfigParser()
    est_init_config.read(est_init_config_filename)
    n_latents = int(est_init_config["model_structure_params"]["n_latents"])
    #    build dynamic parameters
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    dynamic_params = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=args, args_info=args_info)
    #    build configuration file parameters
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=est_init_config).get_dict()
    config_file_params = svGPFA.utils.initUtils.getParamsDictFromStringsDict(
        n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
        args_info=args_info)
    #    build configuration default parameters
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_latents=n_latents)
    #    finally, extract initial parameters from the dynamic
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
    svEM = svGPFA.stats.svEM.SVEM_PyTorch()
    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
        svEM.maximize(model=model, optim_params=optim_params,
                      method=optim_method)

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