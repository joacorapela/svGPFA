import sys
import os
import random
import pickle
import argparse
import configparser
import torch

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
                        default=3)
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
    sim_res_filename_pattern = args.sim_res_filename_pattern
    est_init_config_filename_pattern = args.est_init_config_filename_pattern

    # load data
    sim_res_filename = sim_res_filename_pattern.format(sim_res_number)
    with open(sim_res_filename, "rb") as f:
        simRes = pickle.load(f)
    spikes_times = simRes["spikes"]
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])

    # build params_spec
    n_latents = 3
    n_ind_points = [10, 10, 10]
    params_spec = {}

    trials_start_time = 0.0
    trials_end_time = 1.0
    params_spec["data_structure_params"] = {
        "trials_start_times": [trials_start_time for r in range(n_trials)],
        "trials_end_times":   [trials_end_time for r in range(n_trials)],
    }
    var_mean0 = [torch.normal(mean=0, std=1,
                              size=(n_trials, n_ind_points[k], 1),
                              dtype=torch.double)
                 for k in range(n_latents)]

    diag_value = 1e-2
    var_cov0 = [[] for r in range(n_latents)]
    for k in range(n_latents):
        var_cov0[k] = torch.empty((n_trials, n_ind_points[k], n_ind_points[k]),
                                  dtype=torch.double)
        for r in range(n_trials):
            var_cov0[k][r, :, :] = torch.eye(n_ind_points[k],
                                             dtype=torch.double)*diag_value
    params_spec["variational_params0"] = {
        "variational_mean0": var_mean0,
        "variational_cov0":  var_cov0,
    }
    params_spec["embedding_params0"] = {
        "c0": torch.normal(mean=0.0, std=1.0, size=(n_neurons, n_latents),
                           dtype=torch.double),
        "d0":  torch.normal(mean=0.0, std=1.0, size=(n_neurons, 1),
                            dtype=torch.double),
    }
    expQuadK1_lengthscale = 2.9
    expQuadK2_lengthscale = 0.5
    periodK1_lengthscale = 3.1
    periodK1_period = 1.2
    params_spec["kernels_params0"] = {
         "k_types": ["exponentialQuadratic", "exponentialQuadratic", "periodic"],
         "k_params0": [torch.DoubleTensor([expQuadK1_lengthscale]),
                       torch.DoubleTensor([expQuadK2_lengthscale]),
                       torch.DoubleTensor([periodK1_lengthscale,
                                           periodK1_lengthscale]),
                      ],
    }
    params_spec["ind_points_locs_params0"] = {
        "ind_points_locs0": [trials_start_time +
                             (trials_end_time-trials_start_time) *
                             torch.rand(n_trials, n_ind_points[k], 1,
                                        dtype=torch.double)
                             for k in range(n_latents)]
    }
    params_spec["optim_params"] = {
        "n_quad": 200,
        "prior_cov_reg_param": 1e-5,
        #
        "optim_method": "ECM",
        "em_max_iter": 3,
        #
        "estep_estimate": True,
        "estep_max_iter": 20,
        "estep_lr": 1.0,
        "estep_tolerance_grad": 1e-7,
        "estep_tolerance_change": 1e-9,
        "estep_line_search_fn": "strong_wolfe",
        #
        "mstep_embedding_estimate": True,
        "mstep_embedding_max_iter": 20,
        "mstep_embedding_lr": 1.0,
        "mstep_embedding_tolerance_grad": 1e-7,
        "mstep_embedding_tolerance_change": 1e-9,
        "mstep_embedding_line_search_fn": "strong_wolfe",
        #
        "mstep_kernels_estimate": True,
        "mstep_kernels_max_iter": 20,
        "mstep_kernels_lr": 1.0,
        "mstep_kernels_tolerance_grad": 1e-7,
        "mstep_kernels_tolerance_change": 1e-9,
        "mstep_kernels_line_search_fn": "strong_wolfe",
        #
        "mstep_indpointslocs_estimate": True,
        "mstep_indpointslocs_max_iter": 20,
        "mstep_indpointslocs_lr": 1.0,
        "mstep_indpointslocs_tolerance_grad": 1e-7,
        "mstep_indpointslocs_tolerance_change": 1e-9,
        "mstep_indpointslocs_line_search_fn": "strong_wolfe",
        #
        "verbose": True,
    }
    # get the parameters from params_spec
    params, kernels_types = svGPFA.utils.initUtils.getParamsAndKernelsTypes(
        n_trials=n_trials, n_neurons=n_neurons, n_latents=n_latents,
        dynamic_params=params_spec)

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

    model.setInitialParamsAndData(
        measurements=spikes_times,
        initialParams=params["initial_params"],
        eLLCalculationParams=params["quad_params"],
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
