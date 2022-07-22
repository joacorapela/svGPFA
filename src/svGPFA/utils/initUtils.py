
import numpy as np
import torch
import pandas as pd
import svGPFA.stats.kernelsMatricesStore
import svGPFA.utils.miscUtils


def buildFloatListFromStringRep(stringRep):
    float_list = [float(str) for str in stringRep[1:-1].split(", ")]
    return float_list


def getOptimParams(dynamic_params, config_file_params, default_params,
                   optim_params_info=None, section_name="optim_params"):
    if optim_params_info is None:
        optim_params_info = getArgsInfo()["optim_params"]

    tmp_optim_params = {}
    for param_name in optim_params_info:
        param_conv_func = optim_params_info[param_name]
        param_value = getParam(section_name=section_name,
                               param_name=param_name,
                               dynamic_params=dynamic_params,
                               config_file_params=config_file_params,
                               default_params=default_params,
                               conversion_func=param_conv_func)
        tmp_optim_params[param_name] = param_value
    optim_params = {
        "em_max_iter": tmp_optim_params["em_max_iter"],
        "optim_method": tmp_optim_params["optim_method"],
        "prior_cov_reg_param": tmp_optim_params["prior_cov_reg_param"],
        # estep_
        "estep_estimate": tmp_optim_params["estep_estimate"],
        "estep_optim_params": {
            "max_iter": tmp_optim_params["estep_max_iter"],
            "lr": tmp_optim_params["estep_lr"],
            "tolerance_grad": tmp_optim_params["estep_tolerance_grad"],
            "tolerance_change": tmp_optim_params["estep_tolerance_change"],
            "line_search_fn": tmp_optim_params["estep_line_search_fn"],
        },
        # mstep_embedding_
        "mstep_embedding_estimate": tmp_optim_params["mstep_embedding_estimate"],
        "mstep_embedding_optim_params": {
            "max_iter": tmp_optim_params["mstep_embedding_max_iter"],
            "lr": tmp_optim_params["mstep_embedding_lr"],
            "tolerance_grad": tmp_optim_params["mstep_embedding_tolerance_grad"],
            "tolerance_change": tmp_optim_params["mstep_embedding_tolerance_change"],
            "line_search_fn": tmp_optim_params["mstep_embedding_line_search_fn"],
        },
        # mstep_kernels_
        "mstep_kernels_estimate": tmp_optim_params["mstep_kernels_estimate"],
        "mstep_kernels_optim_params": {
            "max_iter": tmp_optim_params["mstep_kernels_max_iter"],
            "lr": tmp_optim_params["mstep_kernels_lr"],
            "tolerance_grad": tmp_optim_params["mstep_kernels_tolerance_grad"],
            "tolerance_change": tmp_optim_params["mstep_kernels_tolerance_change"],
            "line_search_fn": tmp_optim_params["mstep_kernels_line_search_fn"],
        },
        # mstep_indpointslocs_
        "mstep_indpointslocs_estimate": tmp_optim_params["mstep_indpointslocs_estimate"],
        "mstep_indpointslocs_optim_params": {
            "max_iter": tmp_optim_params["mstep_indpointslocs_max_iter"],
            "lr": tmp_optim_params["mstep_indpointslocs_lr"],
            "tolerance_grad": tmp_optim_params["mstep_indpointslocs_tolerance_grad"],
            "tolerance_change": tmp_optim_params["mstep_indpointslocs_tolerance_change"],
            "line_search_fn": tmp_optim_params["mstep_indpointslocs_line_search_fn"],
        },
    }
    return optim_params


def getDefaultParamsDict(n_neurons, n_latents=3):
    params_dict = {
        "model_structure_params": {"n_latents", n_latents},
        "data_structure_params": {"trials_start_time": 0.0,
                                  "trials_end_time": 1.0},
        "variational_params0": {
            "variational_means0": torch.zeros(n_latents, dtype=torch.double),
            "variational_covs0": 1e-2*torch.diag(torch.tensor([1.0]*n_latents)),
        },
        "embedding_params0": {
            "c0": torch.normal(mean=0.0, std=1.0, size=(n_neurons, n_latents)),
            "d0": torch.normal(mean=0.0, std=1.0, size=(n_neurons, 1)),
        },
        "kernels_params0": {"k_type": "exponentialQuadratic",
                           "k_lengthscale0": 1.0},
        "ind_points_params0": {"n_ind_points": 10,
                              "ind_points_locs0_layout": "equispaced"},
        "optim_params": {"n_quad": 200,
                         "prior_cov_reg_param": 1e-3,
                         "optim_method": "ecm",
                         "em_max_iter": 200,
                         "verbose": True,
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
                        }
    }
    return params_dict


def strTo1DDoubleTensor(aString, sep=" ", dtype=float):
    an_np_array = np.fromstring(aString, sep=sep, dtype=dtype)
    a_tensor = torch.from_numpy(an_np_array)
    return a_tensor


def strTo2DDoubleTensor(aString):
    an_np_matrix = np.matrix(aString)
    a_tensor = torch.from_numpy(an_np_matrix)
    return a_tensor

def getArgsInfo():
    args_info = {"model_structure_params": {"n_latents": int},
                 "data_structure_params": {"trials_start_time": float,
                                           "trials_end_time": float,
                                           "trials_start_times": strTo1DDoubleTensor,
                                           "trials_end_times": strTo1DDoubleTensor},
                 "variational_params0": {"variational_means0": strTo1DDoubleTensor,
                                         "variational_covs0": strTo2DDoubleTensor,
                                         "variational_means0_filename": str,
                                         "variational_covs0_filename": str,
                                         "variational_mean0_filename_latent{:d}_trial{:d}": str,
                                         "variational_cov0_filename_latent{:d}_trial{:d}": str},
                 "embedding_params0": {"c0": strTo2DDoubleTensor,
                                       "d0": strTo2DDoubleTensor,
                                       "c0_filename": str,
                                       "d0_filename": str},
                 "kernels_params0": {"k_type": str,
                                     "k_lengthscale0": float,
                                     "k_period0": float,
                                     "k_type_latent{:d}": str,
                                     "k_lengthscale0_latent{:d}": float,
                                     "k_period0_latent{:d}": float},
                 "ind_points_params0": {"n_ind_points": int,
                                        "ind_points_locs0_layout": str,
                                        "ind_points_locs0_filename": str,
                                        "ind_points_locs0_filename_latent{:d}_trial{:d}": str},
                 "optim_params": {"n_quad": int,
                                  "prior_cov_reg_param": float,
                                  "optim_method": str,
                                  "em_max_iter": int,
                                  "verbose": bool,
                                  #
                                  "estep_estimate": bool,
                                  "estep_max_iter": int,
                                  "estep_lr": float,
                                  "estep_tolerance_grad": float,
                                  "estep_tolerance_change": float,
                                  "estep_line_search_fn": str,
                                  #
                                  "mstep_embedding_estimate": bool,
                                  "mstep_embedding_max_iter": int,
                                  "mstep_embedding_lr": float,
                                  "mstep_embedding_tolerance_grad": float,
                                  "mstep_embedding_tolerance_change": float,
                                  "mstep_embedding_line_search_fn": str,
                                  #
                                  "mstep_kernels_estimate": bool,
                                  "mstep_kernels_max_iter": int,
                                  "mstep_kernels_lr": float,
                                  "mstep_kernels_tolerance_grad": float,
                                  "mstep_kernels_tolerance_change": float,
                                  "mstep_kernels_line_search_fn": str,
                                  #
                                  "mstep_indpointslocs_estimate": bool,
                                  "mstep_indpointslocs_max_iter": int,
                                  "mstep_indpointslocs_lr": float,
                                  "mstep_indpointslocs_tolerance_grad": float,
                                  "mstep_indpointslocs_tolerance_change": float,
                                  "mstep_indpointslocs_line_search_fn": str},
                }
    return args_info

def getParamsDictFromArgs(n_latents, n_trials, args, args_info):
    params_dict = {}
    for key1 in args_info:
        params_dict[key1] = {}
        for key2 in args_info[key1]:
            conversion_func = args_info[key1][key2]
            if "_latent{:d}" in key2:
                for k in range(n_latents):
                    if "_trial{:d}" in key2:
                        for r in range(n_trials):
                            # latent{:d} should appear before _trial{:d} in the
                            # argument label
                            arg_name = key2.format(k, r)
                            if arg_name in args:
                                params_dict[key1][arg_name] = conversion_func(args[arg_name])
                    else:
                        arg_name = key2.format(k)
                        if arg_name in args:
                            params_dict[key1][arg_name] = conversion_func(args[arg_name])
            else:
                arg_name = key2
                if arg_name in args:
                    params_dict[key1][arg_name] = conversion_func(args[arg_name])
    return params_dict


def getParamsDictFromStringsDict(n_latents, n_trials, strings_dict, args_info):
    params_dict = {}
    for key1 in args_info:
        params_dict[key1] = {}
        for key2 in args_info[key1]:
            conversion_func = args_info[key1][key2]
            if "_latent{:d}" in key2:
                for k in range(n_latents):
                    if "_trial{:d}" in key2:
                        for r in range(n_trials):
                            # latent{:d} should appear before _trial{:d} in the
                            # argument label
                            arg_name = key2.format(k, r)
                            if key1 in strings_dict and \
                               arg_name in strings_dict[key1]:
                                params_dict[key1][arg_name] = conversion_func(strings_dict[key1][arg_name])
                    else:
                        arg_name = key2.format(k)
                        if key1 in strings_dict and \
                           arg_name in strings_dict[key1]:
                            params_dict[key1][arg_name] = conversion_func(strings_dict[key1][arg_name])
            else:
                arg_name = key2
                if key1 in strings_dict and \
                   arg_name in strings_dict[key1]:
                    params_dict[key1][arg_name] = conversion_func(strings_dict[key1][arg_name])
    return params_dict


def getParams(n_neurons, n_trials,
              dynamic_params, config_file_params, default_params):

    n_latents = getParam(section_name="model_structure_params",
                         param_name="n_latents",
                         dynamic_params=dynamic_params,
                         config_file_params=config_file_params,
                         default_params=default_params,
                         conversion_func=int)
    n_quad = getParam(section_name="optim_params",
                      param_name="n_quad",
                      dynamic_params=dynamic_params,
                      config_file_params=config_file_params,
                      default_params=default_params,
                      conversion_func=int)
    n_ind_points = getParam(section_name="ind_points_params0",
                            param_name="n_ind_points",
                            dynamic_params=dynamic_params,
                            config_file_params=config_file_params,
                            default_params=default_params,
                            conversion_func=int)

    C0, d0 = getLinearEmbeddingParams0(
        n_neurons=n_neurons, n_latents=n_latents,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)

    trials_start_times, trials_end_times = getTrialsStartEndTimes(
        n_trials=n_trials,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)

    legQuadPoints, legQuadWeights = \
        svGPFA.utils.miscUtils.getLegQuadPointsAndWeights(
            nQuad=n_quad, trials_start_times=trials_start_times,
            trials_end_times=trials_end_times)

    kernels_params0, kernels_types = \
        getKernelsParams0AndTypes(
            n_latents=n_latents,
            dynamic_params=dynamic_params,
            config_file_params=config_file_params,
            default_params=default_params)

    ind_points_locs0 = getIndPointsLocs0(
        n_latents=n_latents, n_trials=n_trials,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params,
        n_ind_points=n_ind_points,
        trials_start_times=trials_start_times,
        trials_end_times=trials_end_times,
    )

    var_mean0 = getVariationalMean0(
        n_latents=n_latents, n_trials=n_trials, n_ind_points=n_ind_points,
        dynamic_params=dynamic_params, config_file_params=config_file_params,
        default_params=default_params)

    var_cov0 = getVariationalCov0(
        n_latents=n_latents, n_trials=n_trials, dynamic_params=dynamic_params,
        config_file_params=config_file_params, default_params=default_params,
        n_ind_points=n_ind_points)
    var_cov0_chol = [svGPFA.utils.miscUtils.chol3D(var_cov0[k])
                     for k in range(n_latents)]
    var_cov0_chol_vecs = \
        svGPFA.utils.miscUtils.getVectorRepOfLowerTrianMatrices(
            lt_matrices=var_cov0_chol)

    optim_params = getOptimParams(
        dynamic_params=dynamic_params, config_file_params=config_file_params,
        default_params=default_params,
    )
    variational_params0 = {"mean": var_mean0,
                           "cholVecs": var_cov0_chol_vecs}
    kmsParams0 = {"kernelsParams0": kernels_params0,
                  "inducingPointsLocs0": ind_points_locs0}
    posteriorLatentsParams0 = {"svPosteriorOnIndPoints": variational_params0,
                               "kernelsMatricesStore": kmsParams0}
    embeddingParams0 = {"C0": C0, "d0": d0}
    initialParams = {"svPosteriorOnLatents": posteriorLatentsParams0,
                     "svEmbedding": embeddingParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    return initialParams, quadParams, kernels_types, optim_params


def getParam(section_name, param_name,
             dynamic_params, config_file_params, default_params,
             conversion_func):
    # dynamic_params
    if dynamic_params is not None and \
            section_name in dynamic_params and \
            param_name in dynamic_params[section_name]:
        param = conversion_func(dynamic_params[section_name][param_name])
        print(f"Extracted dynamic_params[{section_name}][{param_name}]={param}")
    # config_file_params
    elif config_file_params is not None and \
            section_name in config_file_params and \
            param_name in config_file_params[section_name]:
        param = conversion_func(config_file_params[section_name][param_name])
        print(f"Extracted config_file_params[{section_name}][{param_name}]={param}")
    # default_params
    elif default_params is not None and \
            section_name in default_params and \
            param_name in default_params[section_name]:
        param = conversion_func(default_params[section_name][param_name])
        print(f"Extracted default_params[{section_name}][{param_name}]={param}")
    else:
        param = None
    return param


def getLinearEmbeddingParams0(n_neurons, n_latents, dynamic_params,
                              config_file_params, default_params):
    C = getLinearEmbeddingParam0(param_label="c0", n_rows=n_neurons,
                                 n_cols=n_latents,
                                 dynamic_params=dynamic_params,
                                 config_file_params=config_file_params,
                                 default_params=default_params)
    d = getLinearEmbeddingParam0(param_label="d0", n_rows=n_neurons, n_cols=1,
                                 dynamic_params=dynamic_params,
                                 config_file_params=config_file_params,
                                 default_params=default_params)
    C = C.contiguous()
    d = d.contiguous()
    return C, d


def getLinearEmbeddingParam0(param_label, n_rows, n_cols, dynamic_params,
                             config_file_params, default_params):
    if dynamic_params is not None:
        param = getLinearEmbeddingParam0InDict(param_label=param_label,
                                               params_dict=dynamic_params,
                                               params_dict_type="dynamic",
                                               n_rows=n_rows, n_cols=n_cols)
        if param is not None:
            return param

    if config_file_params is not None:
        param = getLinearEmbeddingParam0InDict(param_label=param_label,
                                               params_dict=config_file_params,
                                               params_dict_type="config_file",
                                               n_rows=n_rows, n_cols=n_cols)
        if param is not None:
            return param

    if default_params is not None:
        param = getLinearEmbeddingParam0InDict(param_label=param_label,
                                               params_dict=default_params,
                                               params_dict_type="default",
                                               n_rows=n_rows, n_cols=n_cols)
        if param is not None:
            return param

    raise ValueError("embedding_params not found")


def getLinearEmbeddingParam0InDict(param_label, params_dict,
                                   params_dict_type, n_rows, n_cols,
                                   section_name="embedding_params0"):
    # binary
    if section_name in params_dict and \
       f"{param_label}" in params_dict[section_name]: 
        param = params_dict[section_name][param_label]
        print(f"Extracted {param_label} from {params_dict_type}")
    # filename
    elif section_name in params_dict and \
       f"{param_label}_filename" in params_dict[section_name]: 
        param_filename = params_dict[section_name][f"{param_label}_filename"]
        df = pd.read_csv(param_filename, header=None)
        param = torch.from_numpy(df.values).type(torch.double)
        print(f"Extracted {param_label}_filename from {params_dict_type}")
    # random
    elif section_name in params_dict and \
            f"{param_label}_distribution" in params_dict[section_name] and \
            f"{param_label}_loc" in params_dict[section_name] and \
            f"{param_label}_scale" in params_dict[section_name]:
        param_distribution = params_dict[section_name]\
                                        [f"{param_label}_distribution"]
        param_loc = params_dict[section_name][f"{param_label}_loc"]
        param_scale = params_dict[section_name][f"{param_label}_scale"]
        if f"{param_label}_random_seed" in params_dict[section_name]:
            param_random_seed = params_dict[section_name]\
                                           [f"{param_label}_random_seed"]
        else:
            param_random_seed = None
        print(f"Extracted {param_label}_distribution={param_distribution}, "
              f"{param_label}_loc={param_loc}, "
              f"{param_label}_scale={param_scale}, "
              f"{param_label}_random_seed={param_random_seed} "
              f"from {params_dict_type}")
        # If param_random_seed was specified for replicability
        if param_random_seed is not None:
            torch.random.manual_seed(param_random_seed)
        if param_distribution == "Normal":
            param = torch.distributions.normal.Normal(param_loc, param_scale).sample(sample_shape=[n_rows, n_cols]).type(torch.double)
        else:
            raise ValueError(f"Invalid param_distribution={param_distribution}")
        # If param_random_seed was specified for replicability
        if param_random_seed is not None:
            torch.random.seed()
    else:
        param = None

    return param


def getTrialsStartEndTimes(n_trials, dynamic_params, config_file_params,
                           default_params):
    trials_start_times = getTrialsTimes(
        param_float_label="trials_start_time",
        param_list_label="trials_start_times",
        n_trials=n_trials,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    trials_end_times = getTrialsTimes(
        param_float_label="trials_end_time",
        param_list_label="trials_end_times",
        n_trials=n_trials,
        dynamic_params=dynamic_params,
        config_file_params=config_file_params,
        default_params=default_params)
    return trials_start_times, trials_end_times


def getTrialsTimes(param_list_label, param_float_label, n_trials,
                   dynamic_params, config_file_params, default_params,
                   trials_section_name="data_structure_params"):
    if dynamic_params is not None:
        param = getTrialsTimesInDict(n_trials=n_trials,
                                     param_list_label=param_list_label,
                                     param_float_label=param_float_label,
                                     params_dict=dynamic_params,
                                     params_dict_type="dynamic")
        if param is not None:
            return param

    if config_file_params is not None:
        param = getTrialsTimesInDict(n_trials=n_trials,
                                     param_list_label=param_list_label,
                                     param_float_label=param_float_label,
                                     params_dict=config_file_params,
                                     params_dict_type="config_file")
        if param is not None:
            return param

    if default_params is not None:
        param = getTrialsTimesInDict(n_trials=n_trials,
                                     param_list_label=param_list_label,
                                     param_float_label=param_float_label,
                                     params_dict=default_params,
                                     params_dict_type="default")
        if param is not None:
            return param

    raise ValueError("trials_times not found")


def getTrialsTimesInDict(n_trials, param_list_label, param_float_label,
                         params_dict, params_dict_type,
                         section_name="data_structure_params"):
    if section_name in params_dict and \
       param_list_label in params_dict[section_name]:
        trials_times = strTo1DDoubleTensor(aString=params_dict[section_name][param_list_label])
        print(f"Extracted {param_list_label} from {params_dict_type}")
    elif section_name in params_dict and \
       param_float_label in params_dict[section_name]:
        trials_times_list = [float(params_dict[section_name][param_float_label]) for r in range(n_trials)]
        trials_times = torch.DoubleTensor(trials_times_list)
        print(f"Extracted {param_list_label} from {params_dict_type}")
    else:
        trials_times = None
    return trials_times


def getKernelsParams0AndTypes(n_latents,
                              dynamic_params, config_file_params, default_params):
    if dynamic_params is not None:
        params0, kernels_types = getKernelsParams0AndTypesInDict(
            n_latents=n_latents, params_dict=dynamic_params,
            params_dict_type="dynamic")
        if params0 is not None and kernels_types is not None:
            return params0, kernels_types

    if config_file_params is not None:
        params0, kernels_types = getKernelsParams0AndTypesInDict(
            n_latents=n_latents, params_dict=config_file_params,
            params_dict_type="config_file")
        if params0 is not None and kernels_types is not None:
            return params0, kernels_types

    if default_params is not None:
        params0, kernels_types = getKernelsParams0AndTypesInDict(
            n_latents=n_latents, params_dict=default_params,
            params_dict_type="default")
        if params0 is not None and kernels_types is not None:
            return params0, kernels_types

    raise ValueError("kernels parameters not found")


def getKernelsParams0AndTypesInDict(n_latents, params_dict, params_dict_type,
                                    section_name="kernels_params0"):
    # binary format
    if section_name in params_dict and \
       "k_types" in params_dict[section_name] and \
       "k_params0" in params_dict[section_name]:
        kernels_types = params_dict[section_name]["k_types"]
        params0 = params_dict[section_name]["k_params0"]
        print(f"Extracted k_types={kernels_types} and "
              f"k_params0={params0} from {params_dict_type}")
    # short format
    elif section_name in params_dict and \
       "k_type" in params_dict[section_name]:
        if params_dict[section_name]["k_type"] == "exponentialQuadratic":
            kernels_types = ["exponentialQuadratic" for k in range(n_latents)]
            if "k_lengthscale0" in params_dict[section_name]:
                lengthscale0 = float(params_dict[section_name]["k_lengthscale0"])
            else:
                raise ValueError("If k_type=exponentialQuadratic is specified "
                                 f"in {params_dict_type}, then k_lengthscale0 "
                                 "should also be specified in "
                                 f"{params_dict_type}")
            params0 = [torch.DoubleTensor([lengthscale0])
                       for k in range(n_latents)]
            print("Extracted k_type=exponentialQuadratic and "
                  f"k_lengthsale0={lengthscale0} from {params_dict_type}")
        elif params_dict[section_name]["k_type"] == "periodic":
            kernels_types = ["periodic" for k in range(n_latents)]
            if "k_lengthscale0" in params_dict[section_name]:
                lengthscale0 = float(params_dict[section_name]["k_lengthscale0"])
            else:
                raise ValueError("If k_type=periodic is specified "
                                 f"in {params_dict_type}, then k_lengthscale0 "
                                 "should also be specified in "
                                 f"{params_dict_type}")
            if "k_period0" in params_dict[section_name]:
                period0 = float(params_dict[section_name]["k_period0"])
            else:
                raise ValueError("If k_type=periodic is specified "
                                 f"in {params_dict_type}, then k_period0 "
                                 "should also be specified in "
                                 f"{params_dict_type}")
            params0 = [torch.DoubleTensor([lengthscale0, period0])
                       for k in range(n_latents)]
            print(f"Extracted k_type=periodic, k_lengthsale0={lengthscale0} "
                  f"and  k_period={period0} from {params_dict_type}")
    # long format
    elif section_name in params_dict and \
            "k_type_latent0" in params_dict[section_name]:
        kernels_types = []
        params0 = []
        for k in range(n_latents):
            if params_dict[section_name][f"k_type_latent{k}"] == "exponentialQuadratic":
                kernels_types.append("exponentialQuadratic")
                if f"k_lengthscale0_latent{k}" in params_dict[section_name]:
                    lengthscale0 = \
                            float(params_dict[section_name][f"k_lengthscale0_latent{k}"])
                else:
                    raise ValueError("If k_type=exponentialQuadratic is "
                                     "specified in {params_dict_type}, "
                                     f"then k_lengthscale0_latent{k} "
                                     "should also be specified in "
                                     f"{params_dict_type}")
                params0.append(torch.DoubleTensor([lengthscale0]))
                print(f"Extracted k_type_latent{k}=exponentialQuadratic and "
                      f"k_lengthsale0_latent0{k}={lengthscale0} {params_dict_type}")
            elif params_dict[section_name][f"k_type_latent{k}"] == "periodic":
                kernels_types.append("periodic")
                if "k_lengthscale0_latent{k}" in params_dict[section_name]:
                    lengthscale0 = \
                            float(params_dict[section_name][f"k_lengthscale0_latent{k}"])
                else:
                    raise ValueError("If k_type=periodic is "
                                     "specified in {params_dict_type}, "
                                     f"then k_lengthscale0_latent{k} "
                                     "should also be specified in "
                                     f"{params_dict_type}")
                if "k_period0_latent{k}" in params_dict[section_name]:
                    period0 = \
                            float(params_dict[section_name][f"k_period0_latent{k}"])
                else:
                    raise ValueError("If k_type=periodic is "
                                     f"specified in {params_dict_type}, "
                                     f"then k_period0_latent{k} "
                                     "should also be specified in "
                                     f"{params_dict_type}")
                params0.append(torch.DoubleTensor([lengthscale0, period0]))
                print(f"Extracted k_type_latent{k}=periodic, "
                      f"k_lengthsale0_latent{k}={lengthscale0} and "
                      f"k_period0_latent{k}={period0} from "
                      f"{params_dict_type}")
            else:
                raise RuntimeError("Invalid k_type_latent{:d}={:s}".format(
                    k, params_dict[section_name][f"k_type_latent{k}"]))
    else:
        params0 = None
        kernels_types = None
    return params0, kernels_types


def getIndPointsLocs0(n_latents, n_trials,
                      dynamic_params, config_file_params, default_params,
                      n_ind_points=-1,
                      trials_start_times=None,
                      trials_end_times=None):
    if dynamic_params is not None:
        param = getIndPointsLocs0InDict(n_latents=n_latents, n_trials=n_trials,
                                        params_dict=dynamic_params,
                                        params_dict_type="dynamic",
                                        n_ind_points=n_ind_points,
                                        trials_start_times=trials_start_times,
                                        trials_end_times=trials_end_times)
        if param is not None:
            return param

    if config_file_params is not None:
        param = getIndPointsLocs0InDict(n_latents=n_latents, n_trials=n_trials,
                                        params_dict=config_file_params,
                                        params_dict_type="config_file",
                                        n_ind_points=n_ind_points,
                                        trials_start_times=trials_start_times,
                                        trials_end_times=trials_end_times)
        if param is not None:
            return param

    if default_params is not None:
        param = getIndPointsLocs0InDict(n_latents=n_latents, n_trials=n_trials,
                                        params_dict=default_params,
                                        params_dict_type="default",
                                        n_ind_points=n_ind_points,
                                        trials_start_times=trials_start_times,
                                        trials_end_times=trials_end_times)
        if param is not None:
            return param

    raise ValueError("ind_points_loc0 not found")


def getIndPointsLocs0InDict(n_latents, n_trials, params_dict, params_dict_type,
                            n_ind_points, trials_start_times, trials_end_times,
                            section_name="ind_points_params0"):
    # binary
    if section_name in params_dict and \
       "ind_points_locs0" in params_dict[section_name]:
        ind_points_locs0 = params_dict[section_name]["ind_points_locs0"]
        print(f"Extracted ind_points_locs from {params_dict_type}")
    # filename: same inducing points across all latents and trials
    elif section_name in params_dict and \
       "ind_points_locs0_filename" in params_dict[section_name]:
        ind_points_locs0_filename = \
            params_dict[section_name]["ind_points_locs0_filename"]
        ind_points_locs0 = getSameAcrossLatentsAndTrialsIndPointsLocs0(
            n_latents=n_latents, n_trials=n_trials,
            ind_points_locs0_filename=ind_points_locs0_filename)
        print(f"Extracted ind_points_locs0_filename={ind_points_locs0_filename}"
              f"from {params_dict_type}")
    # filename: different inducing points across all latents and trials
    elif section_name in params_dict and \
            "ind_points_locs0_filename_latent0_trial0" in \
            params_dict[section_name]:
        ind_points_locs0 = getDiffAcrossLatentsAndTrialsIndPointsLocs0(
            n_latents=n_latents, n_trials=n_trials, params_dict=params_dict,
            params_dict_type=params_dict_type)
    # layout
    elif section_name in params_dict and \
            "ind_points_locs0_layout" in params_dict[section_name] and \
            n_ind_points > 0 and  \
            trials_start_times is not None and \
            trials_end_times is not None:
        layout = params_dict[section_name]["ind_points_locs0_layout"]
        print(f"Extracted ind_points_locs0_layout={layout} from "
              f"{params_dict_type}")
        if layout == "equidistant":
            ind_points_locs0 = buildEquidistantIndPointsLocs0(
                n_latents=n_latents, n_trials=n_trials,
                n_ind_points=n_ind_points,
                trials_start_times=trials_start_times,
                trials_end_times=trials_end_times)
        else:
            raise RuntimeError(f"Invalid ind_points_locs0_layout={layout}")
    else:
        ind_points_locs0 = None

    return ind_points_locs0


def getSameAcrossLatentsAndTrialsIndPointsLocs0(
        n_latents, n_trials, ind_points_locs0_filename):
    Z0 = torch.from_numpy(pd.read_csv(ind_points_locs0_filename,
                                      header=None).to_numpy()).flatten()
    Z0s = [[] for k in range(n_latents)]
    nIndPointsForLatent = len(Z0)
    for k in range(n_latents):
        Z0s[k] = torch.empty((n_trials, nIndPointsForLatent, 1),
                             dtype=torch.double)
        Z0s[k][:, :, 0] = Z0
    return Z0s


def getDiffAcrossLatentsAndTrialsIndPointsLocs0(
        n_latents, n_trials, params_dict, params_dict_type,
        section_name="ind_points_params0",
        item_name_pattern="ind_points_locs0_filename_latent{:d}_trial{:d}"):
    Z0 = [[] for k in range(n_latents)]
    for k in range(n_latents):
        item_name = item_name_pattern.format(k, 0)
        ind_points_locs0_filename = params_dict[section_name][item_name]
        print(f"Extracted {item_name}={ind_points_locs0_filename} from "
              f"{params_dict_type}")
        Z0_k_r0 = torch.from_numpy(pd.read_csv(ind_points_locs0_filename, header=None).to_numpy()).flatten()
        nIndPointsForLatent = len(Z0_k_r0)
        Z0[k] = torch.empty((n_trials, nIndPointsForLatent, 1),
                            dtype=torch.double)
        Z0[k][0, :, 0] = Z0_k_r0
        for r in range(1, n_trials):
            item_name = item_name_pattern.format(k, r)
            ind_points_locs0_filename = params_dict[section_name][item_name]
            print(f"Extracted {item_name}={ind_points_locs0_filename} from "
                  f"{params_dict_type}")
            Z0_k_r = torch.from_numpy(pd.read_csv(ind_points_locs0_filename, header=None).to_numpy()).flatten()
            Z0[k][r, :, 0] = Z0_k_r
    return Z0


def buildEquidistantIndPointsLocs0(n_latents, n_trials, n_ind_points,
                                   trials_start_times, trials_end_times):
    Z0s = [[] for k in range(n_latents)]
    for k in range(n_latents):
        Z0s[k] = torch.empty((n_trials, n_ind_points, 1), dtype=torch.double)
        for r in range(n_trials):
            Z0 = torch.linspace(trials_start_times[r], trials_end_times[r],
                                n_ind_points)
            Z0s[k][r, :, 0] = Z0
    return Z0s


def getVariationalMean0(n_latents, n_trials, n_ind_points,
                        dynamic_params, config_file_params, default_params):
    if dynamic_params is not None:
        param = getVariationalMean0InDict(n_latents=n_latents,
                                          n_trials=n_trials,
                                          n_ind_points=n_ind_points,
                                          params_dict=dynamic_params,
                                          params_dict_type="dynamic")
        if param is not None:
            return param

    if config_file_params is not None:
        param = getVariationalMean0InDict(n_latents=n_latents,
                                          n_trials=n_trials,
                                          n_ind_points=n_ind_points,
                                          params_dict=config_file_params,
                                          params_dict_type="config_file")
        if param is not None:
            return param

    if default_params is not None:
        param = getVariationalMean0InDict(n_latents=n_latents,
                                          n_trials=n_trials,
                                          n_ind_points=n_ind_points,
                                          params_dict=default_params,
                                          params_dict_type="default")
        if param is not None:
            return param

    raise ValueError("variational_mean0 not found")


def getVariationalMean0InDict(n_latents, n_trials, n_ind_points,
                              params_dict, params_dict_type,
                              section_name="variational_params0",
                              binary_item_name="variational_mean0",
                              common_filename_item_name=
                               "variational_means0_filename",
                              different_filename_item_name_pattern=
                               "variational_mean0_filename_latent{:d}_trial{:d}",
                              constant_value_item_name=
                               "variational_mean0_constant_value"):
    # binary
    if section_name in params_dict and \
       binary_item_name in params_dict[section_name]:
        variational_mean0 = params_dict[section_name][binary_item_name]
        print("Extracted "
              f"{binary_item_name} from "
              f"{params_dict_type}")
    # variational_means_filename
    elif section_name in params_dict and \
            common_filename_item_name in params_dict[section_name]:
        variational_mean0_filename = \
            params_dict[section_name][common_filename_item_name]
        print("Extracted "
              f"{common_filename_item_name}={variational_mean0_filename} from "
              f"{params_dict_type}")
        a_variational_mean0 = torch.from_numpy(pd.read_csv(variational_mean0_filename, header=None).to_numpy()).flatten()
        variational_mean0 = getSameAcrossLatentsAndTrialsVariationalMean0(
            n_latents=n_latents, n_trials=n_trials,
            a_variational_mean0=a_variational_mean0)
    # variational_means_filename latent k trial r
    elif section_name in params_dict and \
            different_filename_item_name_pattern.format(0, 0) in params_dict[section_name]:
        variational_mean0 = getDiffAcrossLatentsAndTrialsVariationalMean0(
            n_latents=n_latents, n_trials=n_trials, params_dict=params_dict,
            params_dict_type=params_dict_type,
            section_name=section_name,
            item_name_pattern=different_filename_item_name_pattern)
    # constant_value
    elif section_name in params_dict and \
            constant_value_item_name in params_dict[section_name]:
        constant_value = params_dict[section_name][constant_value_item_name]
        print(f"Extracted {constant_value_item_name}={constant_value} from "
              f"{params_dict_type}")
        a_variational_mean0 = constant_value * torch.ones(n_ind_points, dtype=torch.double)
        variational_mean0 = getSameAcrossLatentsAndTrialsVariationalMean0(
            n_latents=n_latents, n_trials=n_trials,
            a_variational_mean0=a_variational_mean0)
    else:
        variational_mean0 = None
    return variational_mean0


def getSameAcrossLatentsAndTrialsVariationalMean0(n_latents, n_trials,
                                                  a_variational_mean0):
    variational_mean0 = [[] for r in range(n_latents)]
    nIndPoints = len(a_variational_mean0)
    for k in range(n_latents):
        variational_mean0[k] = torch.empty((n_trials, nIndPoints, 1), dtype=torch.double)
        variational_mean0[k][:, :, 0] = a_variational_mean0
    return variational_mean0


def getDiffAcrossLatentsAndTrialsVariationalMean0(
        n_latents, n_trials, params_dict, params_dict_type,
        section_name="variational_params0",
        item_name_pattern="variational_mean_latent{:d}_trial{:d}_filename"):
    variational_mean0 = [[] for r in range(n_latents)]
    for k in range(n_latents):
        variational_mean0_filename = params_dict[section_name][item_name_pattern.format(k, 0)]
        print(f"Extracted {item_name_pattern.format(k, 0)}={variational_mean0_filename} from {params_dict_type}")
        variational_mean0_k0 = torch.from_numpy(pd.read_csv(variational_mean0_filename, header=None).to_numpy()).flatten()
        nIndPointsK = len(variational_mean0_k0)
        variational_mean0[k] = torch.empty((n_trials, nIndPointsK, 1), dtype=torch.double)
        variational_mean0[k][0, :, 0] = variational_mean0_k0
        for r in range(1, n_trials):
            variational_mean0_filename = params_dict[section_name][item_name_pattern.format(k, r)]
            print(f"Extracted {item_name_pattern.format(k, r)}={variational_mean0_filename} from {params_dict_type}")
            variational_mean0_kr = torch.from_numpy(pd.read_csv(variational_mean0_filename, header=None).to_numpy()).flatten()
            variational_mean0[k][r, :, 0] = variational_mean0_kr
    return variational_mean0


def getVariationalCov0(n_latents, n_trials,
                       dynamic_params, config_file_params, default_params,
                       n_ind_points=-1):
    if dynamic_params is not None:
        param = getVariationalCov0InDict(n_latents=n_latents,
                                         n_trials=n_trials,
                                         n_ind_points=n_ind_points,
                                         params_dict=dynamic_params,
                                         params_dict_type="dynamic")
        if param is not None:
            return param

    if config_file_params is not None:
        param = getVariationalCov0InDict(n_latents=n_latents,
                                         n_trials=n_trials,
                                         n_ind_points=n_ind_points,
                                         params_dict=config_file_params,
                                         params_dict_type="config_file")
        if param is not None:
            return param

    if default_params is not None:
        param = getVariationalCov0InDict(n_latents=n_latents,
                                         n_trials=n_trials,
                                         n_ind_points=n_ind_points,
                                         params_dict=default_params,
                                         params_dict_type="default")
        if param is not None:
            return param

    raise ValueError("variationalCov0 not found")


def getVariationalCov0InDict(n_latents, n_trials, params_dict,
                             params_dict_type,
                             n_ind_points=-1,
                             section_name="variational_params0",
                             binary_item_name="variational_cov0",
                             common_filename_item_name="variational_covs0_filename",
                             different_filename_item_name_pattern="variational_cov0_filename_latent{:d}_trial{:d}",
                             diag_value_item_name="variational_cov0_diag_value"):
    # binary variational mean and cov
    if section_name in params_dict and \
       binary_item_name in params_dict[section_name]:
        variational_cov0 = params_dict[section_name][binary_item_name]
        print(f"Extracted {binary_item_name} from {params_dict_type}")
    # diag_value
    elif section_name in params_dict and \
       diag_value_item_name in params_dict[section_name]:
        diag_value = params_dict[section_name][common_filename_item_name]
        print(f"Extracted {diag_value_item_name}={diag_value} from "
              f"{params_dict_type}")
        variational_cov0 = diag_value * torch.eye(n_ind_points, dtype=torch.double)
    # common_filename
    elif section_name in params_dict and \
            common_filename_item_name in params_dict[section_name]:
        variational_cov0_filename =  params_dict[section_name][common_filename_item_name]
        print("Extracted "
              f"{common_filename_item_name}={variational_cov0_filename} from "
              f"{params_dict_type}")
        a_variational_cov0 = torch.from_numpy(pd.read_csv(variational_cov0_filename, header=None).to_numpy())
        variational_cov0 = getSameAcrossLatentsAndTrialsVariationalCov0(
            n_latents=n_latents, n_trials=n_trials,
            a_variational_cov0=a_variational_cov0,
            section_name=section_name)
    # variational_cov0_latent{:d}_trial{:d}_filename
    elif section_name in params_dict and \
            different_filename_item_name_pattern.format(0, 0) in params_dict[section_name]:
        variational_cov0 = getDiffAcrossLatentsAndTrialsVariationalCov0(
            n_latents=n_latents, n_trials=n_trials,
            params_dict=params_dict, params_dict_type=params_dict_type)
    else:
        variational_cov0 = None
    return variational_cov0


def getSameAcrossLatentsAndTrialsVariationalCov0(
        n_latents, n_trials, a_variational_cov0,
        section_name="variational_params0"):
    variational_cov0 = [[] for r in range(n_latents)]
    nIndPoints = a_variational_cov0.shape[0]
    for k in range(n_latents):
        variational_cov0[k] = torch.empty((n_trials, nIndPoints, nIndPoints),
                                          dtype=torch.double)
        variational_cov0[k][:, :, :] = a_variational_cov0
    return variational_cov0


def getDiffAcrossLatentsAndTrialsVariationalCov0(
        n_latents, n_trials, params_dict, params_dict_type,
        section_name="variational_params0",
        item_name_pattern="variational_cov0_filename_latent{:d}_trial{:d}"):
    variational_cov0 = [[] for r in range(n_latents)]
    for k in range(n_latents):
        item_name = item_name_pattern.format(k, 0)
        variational_cov_filename = params_dict[section_name][item_name]
        print(f"Extracted {item_name}={variational_cov_filename} "
              f"from config[{section_name}]")
        variational_cov0_k0 = torch.from_numpy(pd.read_csv(variational_cov_filename, header=None).to_numpy())
        nIndPointsK = variational_cov0_k0.shape[0]
        variational_cov0[k] = torch.empty((n_trials, nIndPointsK, nIndPointsK), dtype=torch.double)
        variational_cov0[k][0, :, :] = variational_cov0_k0
        for r in range(1, n_trials):
            item_name = item_name_pattern.format(k, r)
            variational_cov_filename = params_dict[section_name][item_name]
            print(f"Extracted {item_name}={variational_cov_filename} from config[{section_name}]")
            variational_cov0_kr = torch.from_numpy(pd.read_csv(variational_cov_filename, header=None).values)
            variational_cov0[k][r, :, :] = variational_cov0_kr
    return variational_cov0


def getUniformIndPointsMeans(n_trials, n_latents, nIndPointsPerLatent, min=-1, max=1):
    ind_points_means = [[] for r in range(n_trials)]
    for r in range(n_trials):
        ind_points_means[r] = [[] for k in range(n_latents)]
        for k in range(n_latents):
            ind_points_means[r][k] = torch.rand(nIndPointsPerLatent[k], 1)*(max-min)+min
    return ind_points_means


def getConstantIndPointsMeans(constantValue, n_trials, n_latents, nIndPointsPerLatent):
    ind_points_means = [[] for r in range(n_trials)]
    for r in range(n_trials):
        ind_points_means[r] = [[] for k in range(n_latents)]
        for k in range(n_latents):
            ind_points_means[r][k] = constantValue*torch.ones(nIndPointsPerLatent[k], 1, dtype=torch.double)
    return ind_points_means


def getKzzChol0(kernels, kernelsParams0, ind_points_locs0, epsilon):
    ind_points_locs_kms = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS()
    ind_points_locs_kms.setKernels(kernels=kernels)
    ind_points_locs_kms.setKernelsParams(kernelsParams=kernelsParams0)
    ind_points_locs_kms.setIndPointsLocs(ind_points_locs=ind_points_locs0)
    ind_points_locs_kms.setEpsilon(epsilon=epsilon)
    ind_points_locs_kms.buildKernelsMatrices()
    KzzChol0 = ind_points_locs_kms.getKzzChol()
    return KzzChol0


def getScaledIdentityQSigma0(scale, n_trials, nIndPointsPerLatent):
    nLatent = len(nIndPointsPerLatent)
    qSigma0 = [[None] for k in range(nLatent)]

    for k in range(nLatent):
        qSigma0[k] = torch.empty((n_trials, nIndPointsPerLatent[k], nIndPointsPerLatent[k]), dtype=torch.double)
        for r in range(n_trials):
            qSigma0[k][r,:,:] = scale*torch.eye(nIndPointsPerLatent[k], dtype=torch.double)
    return qSigma0


def getSVPosteriorOnIndPointsParams0(nIndPointsPerLatent, n_latents, n_trials, scale):
    qMu0 = [[] for k in range(n_latents)]
    qSVec0 = [[] for k in range(n_latents)]
    qSDiag0 = [[] for k in range(n_latents)]
    for k in range(n_latents):
        # qMu0[k] = torch.rand(n_trials, nIndPointsPerLatent[k], 1, dtype=torch.double)
        qMu0[k] = torch.zeros(n_trials, nIndPointsPerLatent[k], 1, dtype=torch.double)
        qSVec0[k] = scale*torch.eye(nIndPointsPerLatent[k], 1, dtype=torch.double).repeat(n_trials, 1, 1)
        qSDiag0[k] = scale*torch.ones(nIndPointsPerLatent[k], 1, dtype=torch.double).repeat(n_trials, 1, 1)
    return qMu0, qSVec0, qSDiag0


def getKernelsParams0(kernels, noiseSTD):
    n_latents = len(kernels)
    kernelsParams0 = [kernels[k].getParams() for k in range(n_latents)]
    if noiseSTD > 0.0:
        kernelsParams0 = [kernelsParams0[0] +
                          noiseSTD*torch.randn(len(kernelsParams0[k]))
                          for k in range(n_latents)]
    return kernelsParams0


def getKernelsScaledParams0(kernels, noiseSTD):
    n_latents = len(kernels)
    kernelsParams0 = [kernels[k].getScaledParams() for k in range(n_latents)]
    if noiseSTD > 0.0:
        kernelsParams0 = [kernelsParams0[0] +
                          noiseSTD*torch.randn(len(kernelsParams0[k]))
                          for k in range(n_latents)]
    return kernelsParams0