"""
Example svGPFA script for parameter estimation -- default parameters
====================================================================

"""
import sys
import os
import random
import pickle
import argparse
import configparser

import gcnu_common.utils.config_dict
import gcnu_common.utils.argparse
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.configUtils
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_res_number", help="simuluation result number",
                        type=int, default=32451751)
    parser.add_argument("--est_init_number", help="estimation init number",
                        type=int, default=545)
    parser.add_argument("--n_latents", help="number of latents", type=int,
                        default=2)
    parser.add_argument("--trials_start_time", help="trials start time",
                        type=float, default=0.0)
    parser.add_argument("--trials_end_time", help="trials end time",
                        type=float, default=1.0)
    parser.add_argument("--sim_res_filename_pattern",
                        help="simuluation result filename pattern",
                        type=str, default="../data/{:08d}_simRes.pickle")
    parser.add_argument("--est_init_config_filename_pattern",
                        help="estimation initialization configuration "
                             "filename pattern",
                        type=str,
                        default="../params/{:08d}_estimation_metaData.ini")

    args, remaining = parser.parse_known_args()
    gcnu_common.utils.argparse.add_remaining_to_populated_args(
        populated=args, remaining=remaining)
    sim_res_number = args.sim_res_number
    est_init_number = args.est_init_number
    n_latents = args.n_latents
    trials_start_time = args.trials_start_time
    trials_end_time = args.trials_end_time
    sim_res_filename_pattern = args.sim_res_filename_pattern
    est_init_config_filename_pattern = args.est_init_config_filename_pattern

    # load data
    sim_res_filename = sim_res_filename_pattern.format(sim_res_number)
    with open(sim_res_filename, "rb") as f:
        simRes = pickle.load(f)
    spikes_times = simRes["spikes"]
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    trials_start_times = [trials_start_time] * n_trials
    trials_end_times = [trials_end_time] * n_trials

    #    build dynamic parameter specifications
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    dynamic_params_spec = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=vars(args),
        args_info=args_info)
    #    build configuration file parameter specifications
    est_init_config_filename = est_init_config_filename_pattern.format(
        est_init_number)
    est_init_config = configparser.ConfigParser()
    est_init_config.read(est_init_config_filename)
    strings_dict = gcnu_common.utils.config_dict.GetDict(
        config=est_init_config).get_dict()
    config_file_params_spec = \
        svGPFA.utils.initUtils.getParamsDictFromStringsDict(
            n_latents=n_latents, n_trials=n_trials, strings_dict=strings_dict,
            args_info=args_info)
    #    finally, get the parameters from the dynamic,
    #    and configuration file parameter specifications
    params, kernels_types = svGPFA.utils.initUtils.getParamsAndKernelsTypes(
        n_trials=n_trials, n_neurons=n_neurons, n_latents=n_latents,
        trials_start_times=trials_start_times,
        trials_end_times=trials_end_times,
        dynamic_params_spec=dynamic_params_spec,
        config_file_params_spec=config_file_params_spec)

    # build modelSaveFilename
    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estimResMetaDataFilename = \
            "../results/{:08d}_estimation_metaData.ini".format(estResNumber)
        if not os.path.exists(estimResMetaDataFilename):
            estPrefixUsed = False
    modelSaveFilename = "../results/{:08d}_estimatedModel.pickle".\
        format(estResNumber)

    # build kernels
    kernels_params0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params0)

    # create model
    model = svGPFA.stats.svGPFAModelFactory.SVGPFAModelFactory.\
        buildModelPyTorch(kernels=kernels)

    model.setParamsAndData(
        measurements=spikes_times,
        initial_params=params["initial_params"],
        eLLCalculationParams=params["ell_calculation_params"],
        priorCovRegParam=params["optim_params"]["prior_cov_reg_param"])

    # maximize lower bound
    svEM = svGPFA.stats.svEM.SVEM_PyTorch()
    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
        svEM.maximize(model=model, optim_params=params["optim_params"],
                      method=params["optim_params"]["optim_method"])

    # save estimated values
    estimResConfig = configparser.ConfigParser()
    estimResConfig["simulation_params"] = {"sim_res_number": sim_res_number}
    estimResConfig["optim_params"] = params["optim_params"]
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


if __name__ == "__main__":
    main(sys.argv)
