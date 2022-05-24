
import torch
import pandas as pd
import svGPFA.stats.kernelsMatricesStore
import svGPFA.utils.miscUtils


def buildFloatListFromStringRep(stringRep):
    float_list = [float(str) for str in stringRep[1:-1].split(", ")]
    return float_list


def getOptimParams(args, config=None,
#                    steps=["estep", "mstep_embedding", "mstep_kernels", "mstep_indpointslocs"],
                   em_max_iter_dft=50,
                   optim_method_dft="ECM",
                   estep_estimate_dft="True",
                   estep_max_iter_dft=20,
                   estep_lr_dft=1.0,
                   estep_tolerance_grad_dft=1e-7,
                   estep_tolerance_change_dft=1e-9,
                   estep_line_search_fn_dft="strong_wolfe",
                   mstep_embedding_estimate_dft="True",
                   mstep_embedding_max_iter_dft=20,
                   mstep_embedding_lr_dft=1.0,
                   mstep_embedding_tolerance_grad_dft=1e-7,
                   mstep_embedding_tolerance_change_dft=1e-9,
                   mstep_embedding_line_search_fn_dft="strong_wolfe",
                   mstep_kernels_estimate_dft="True",
                   mstep_kernels_max_iter_dft=20,
                   mstep_kernels_lr_dft=1.0,
                   mstep_kernels_tolerance_grad_dft=1e-7,
                   mstep_kernels_tolerance_change_dft=1e-9,
                   mstep_kernels_line_search_fn_dft="strong_wolfe",
                   mstep_indpointslocs_estimate_dft="True",
                   mstep_indpointslocs_max_iter_dft=20,
                   mstep_indpointslocs_lr_dft=1.0,
                   mstep_indpointslocs_tolerance_grad_dft=1e-7,
                   mstep_indpointslocs_tolerance_change_dft=1e-9,
                   mstep_indpointslocs_line_search_fn_dft="strong_wolfe",
                   verbose_dft="True",
                  ):

    optim_params = {}
    em_max_iter = getIntParam(section_name="optim_params",
                              param_name="em_max_iter",
                              args=args, config=config,
                              default_value=em_max_iter_dft)
    optim_params["em_max_iter"] = em_max_iter

    optim_method = getStringParam(section_name="optim_params",
                                  param_name="optim_method",
                                  args=args, config=config,
                                  default_value=optim_method_dft)
    optim_params["optim_method"] = optim_method

    prior_cov_reg_param = getFloatParam(section_name="optim_params",
                                        param_name="prior_cov_reg_param",
                                        args=args, config=config,
                                        default_value=em_max_iter_dft)
    optim_params["prior_cov_reg_param"] = prior_cov_reg_param

    # estep
    estep_estimate = getStringParam(section_name="optim_params",
                                    param_name="estep_estimate",
                                    args=args, config=config,
                                    default_value=estep_estimate_dft)=="True"
    optim_params["estep_estimate"] = estep_estimate

    estep_max_iter = getIntParam(section_name="optim_params",
                                 param_name="estep_max_iter",
                                 args=args, config=config,
                                 default_value=estep_max_iter_dft)
    estep_lr = getFloatParam(section_name="optim_params",
                             param_name="estep_lr",
                             args=args, config=config,
                             default_value=estep_lr_dft)
    estep_tolerance_grad = getFloatParam(section_name="optim_params",
                                         param_name="estep_tolerance_grad",
                                         args=args, config=config,
                                         default_value=estep_tolerance_grad_dft)
    estep_tolerance_change = getFloatParam(section_name="optim_params",
                                           param_name="estep_tolerance_change",
                                           args=args, config=config,
                                           default_value=estep_tolerance_change_dft)
    estep_line_search_fn = getStringParam(section_name="optim_params",
                                          param_name="estep_line_search_fn",
                                          args=args, config=config,
                                          default_value=estep_line_search_fn_dft)
    optim_params["estep_optim_params"] = {
        "max_iter": estep_max_iter,
        "lr": estep_lr,
        "tolerance_grad": estep_tolerance_grad,
        "tolerance_change": estep_tolerance_change,
        "line_search_fn": estep_line_search_fn,
    }

    # mstep_embedding
    mstep_embedding_estimate = getStringParam(section_name="optim_params",
                                    param_name="mstep_embedding_estimate",
                                    args=args, config=config,
                                    default_value=mstep_embedding_estimate_dft)=="True"
    optim_params["mstep_embedding_estimate"] = mstep_embedding_estimate

    mstep_embedding_max_iter = getIntParam(section_name="optim_params",
                                 param_name="mstep_embedding_max_iter",
                                 args=args, config=config,
                                 default_value=mstep_embedding_max_iter_dft)
    mstep_embedding_lr = getFloatParam(section_name="optim_params",
                             param_name="mstep_embedding_lr",
                             args=args, config=config,
                             default_value=mstep_embedding_lr_dft)
    mstep_embedding_tolerance_grad = getFloatParam(section_name="optim_params",
                                         param_name="mstep_embedding_tolerance_grad",
                                         args=args, config=config,
                                         default_value=mstep_embedding_tolerance_grad_dft)
    mstep_embedding_tolerance_change = getFloatParam(section_name="optim_params",
                                           param_name="mstep_embedding_tolerance_change",
                                           args=args, config=config,
                                           default_value=mstep_embedding_tolerance_change_dft)
    mstep_embedding_line_search_fn = getStringParam(section_name="optim_params",
                                          param_name="mstep_embedding_line_search_fn",
                                          args=args, config=config,
                                          default_value=mstep_embedding_line_search_fn_dft)
    optim_params["mstep_embedding_optim_params"] = {
        "max_iter": mstep_embedding_max_iter,
        "lr": mstep_embedding_lr,
        "tolerance_grad": mstep_embedding_tolerance_grad,
        "tolerance_change": mstep_embedding_tolerance_change,
        "line_search_fn": mstep_embedding_line_search_fn,
    }

    # mstep_kernels
    mstep_kernels_estimate = getStringParam(section_name="optim_params",
                                    param_name="mstep_kernels_estimate",
                                    args=args, config=config,
                                    default_value=mstep_kernels_estimate_dft)=="True"
    optim_params["mstep_kernels_estimate"] = mstep_kernels_estimate

    mstep_kernels_max_iter = getIntParam(section_name="optim_params",
                                 param_name="mstep_kernels_max_iter",
                                 args=args, config=config,
                                 default_value=mstep_kernels_max_iter_dft)
    mstep_kernels_lr = getFloatParam(section_name="optim_params",
                             param_name="mstep_kernels_lr",
                             args=args, config=config,
                             default_value=mstep_kernels_lr_dft)
    mstep_kernels_tolerance_grad = getFloatParam(section_name="optim_params",
                                         param_name="mstep_kernels_tolerance_grad",
                                         args=args, config=config,
                                         default_value=mstep_kernels_tolerance_grad_dft)
    mstep_kernels_tolerance_change = getFloatParam(section_name="optim_params",
                                           param_name="mstep_kernels_tolerance_change",
                                           args=args, config=config,
                                           default_value=mstep_kernels_tolerance_change_dft)
    mstep_kernels_line_search_fn = getStringParam(section_name="optim_params",
                                          param_name="mstep_kernels_line_search_fn",
                                          args=args, config=config,
                                          default_value=mstep_kernels_line_search_fn_dft)
    optim_params["mstep_kernels_optim_params"] = {
        "max_iter": mstep_kernels_max_iter,
        "lr": mstep_kernels_lr,
        "tolerance_grad": mstep_kernels_tolerance_grad,
        "tolerance_change": mstep_kernels_tolerance_change,
        "line_search_fn": mstep_kernels_line_search_fn,
    }

    # mstep_indpointslocs
    mstep_indpointslocs_estimate = getStringParam(section_name="optim_params",
                                    param_name="mstep_indpointslocs_estimate",
                                    args=args, config=config,
                                    default_value=mstep_indpointslocs_estimate_dft)=="True"
    optim_params["mstep_indpointslocs_estimate"] = mstep_indpointslocs_estimate

    mstep_indpointslocs_max_iter = getIntParam(section_name="optim_params",
                                 param_name="mstep_indpointslocs_max_iter",
                                 args=args, config=config,
                                 default_value=mstep_indpointslocs_max_iter_dft)
    mstep_indpointslocs_lr = getFloatParam(section_name="optim_params",
                             param_name="mstep_indpointslocs_lr",
                             args=args, config=config,
                             default_value=mstep_indpointslocs_lr_dft)
    mstep_indpointslocs_tolerance_grad = getFloatParam(section_name="optim_params",
                                         param_name="mstep_indpointslocs_tolerance_grad",
                                         args=args, config=config,
                                         default_value=mstep_indpointslocs_tolerance_grad_dft)
    mstep_indpointslocs_tolerance_change = getFloatParam(section_name="optim_params",
                                           param_name="mstep_indpointslocs_tolerance_change",
                                           args=args, config=config,
                                           default_value=mstep_indpointslocs_tolerance_change_dft)
    mstep_indpointslocs_line_search_fn = getStringParam(section_name="optim_params",
                                          param_name="mstep_indpointslocs_line_search_fn",
                                          args=args, config=config,
                                          default_value=mstep_indpointslocs_line_search_fn_dft)
    optim_params["mstep_indpointslocs_optim_params"] = {
        "max_iter": mstep_indpointslocs_max_iter,
        "lr": mstep_indpointslocs_lr,
        "tolerance_grad": mstep_indpointslocs_tolerance_grad,
        "tolerance_change": mstep_indpointslocs_tolerance_change,
        "line_search_fn": mstep_indpointslocs_line_search_fn,
    }

    # verbose
    verbose = getStringParam(section_name="optim_params",
                             param_name="verbose",
                             args=args, config=config,
                             default_value=verbose_dft)=="True"
    optim_params["verbose"] = verbose
    return optim_params


#     for step in steps:
#         optim_params["{:s}_estimate".format(step)] = \
#             optim_params_config["{:s}_estimate".format(step)] == "True"
#         optim_params["{:s}_optim_params".format(step)] = {
#             "max_iter": int(optim_params_config["{:s}_max_iter".format(step)]),
#             "lr": float(optim_params_config["{:s}_lr".format(step)]),
#             "tolerance_grad": float(optim_params_config["{:s}_tolerance_grad".format(step)]),
#             "tolerance_change": float(optim_params_config["{:s}_tolerance_change".format(step)]),
#             "line_search_fn": optim_params_config["{:s}_line_search_fn".format(step)],
#         }
#     optim_params["verbose"] = optim_params_config["verbose"] == "True"
#     return optim_params


def getInitialAndQuadParamsAndKernelsTypes(
        n_neurons, n_trials, args,
        config=None, n_latents_dft=2, n_quad_dft=200,
        embedding_matrix_distribution_dft="Normal",
        embedding_matrix_loc_dft=0.0,
        embedding_matrix_scale_dft=1.0,
        embedding_offset_distribution_dft="Normal",
        embedding_offset_loc_dft=0.0,
        embedding_offset_scale_dft=1.0,
        trials_start_time_dft=-1.0,
        trials_end_time_dft=-1.0,
        k_type_dft="exponentialQuadratic",
        k_params0_dft=torch.DoubleTensor([1.0]),
        n_ind_points_dft=9, 
        ind_points_locs_layout_dft="equidistant"):

    n_latents = getIntParam(section_name="other_params",
                               param_name="n_latents",
                               args=args, config=config,
                               default_value=n_latents_dft)

    n_quad = getIntParam(section_name="other_params",
                         param_name="n_quad",
                         args=args, config=config,
                         default_value=n_quad_dft)

    n_ind_points = getIntParam(section_name="other_params",
                               param_name="n_ind_points",
                               args=args, config=config,
                               default_value=n_ind_points_dft)

    ind_points_locs_layout = getIntParam(section_name="other_params",
                                         param_name="ind_points_locs_layout",
                                         args=args, config=config,
                                         default_value=ind_points_locs_layout_dft)

    C0, d0 = getLinearEmbeddingParams0(
        n_neurons=n_neurons, n_latents=n_latents, args=args, config=config,
        embedding_matrix_distribution_dft=embedding_matrix_distribution_dft,
        embedding_matrix_loc_dft=embedding_matrix_loc_dft,
        embedding_matrix_scale_dft=embedding_matrix_scale_dft,
        embedding_offset_distribution_dft=embedding_offset_distribution_dft,
        embedding_offset_loc_dft=embedding_offset_loc_dft,
        embedding_offset_scale_dft=embedding_offset_scale_dft,
    )

    trials_start_times, trials_end_times = getTrialsStartEndTimes(
        n_trials=n_trials, args=args, config=config,
        trials_start_time_dft=trials_start_time_dft,
        trials_end_time_dft=trials_end_time_dft)

    legQuadPoints, legQuadWeights = \
        svGPFA.utils.miscUtils.getLegQuadPointsAndWeights(
            nQuad=n_quad, trials_start_times=trials_start_times,
            trials_end_times=trials_end_times)

    kernels_params0, kernels_types = \
        getKernelsParams0AndTypes(
            n_latents=n_latents, args=args, config=config,
            k_type_dft=k_type_dft, k_params0_dft=k_params0_dft)

    ind_points_locs0 = getIndPointsLocs0(
        n_latents=n_latents, n_trials=n_trials,
        n_ind_points=n_ind_points,
        ind_points_locs_layout=ind_points_locs_layout,
        trials_start_times=trials_start_times,
        trials_end_times=trials_end_times,
        args=args, config=config)

    var_mean0 = getVariationalMean0(
        n_latents=n_latents, n_trials=n_trials, args=args, config=config,
        n_ind_points=n_ind_points)
    var_cov0 = getVariationalCov0(
        n_latents=n_latents, n_trials=n_trials, args=args, config=config,
        n_ind_points=n_ind_points)
    var_cov0_chol = [svGPFA.utils.miscUtils.chol3D(var_cov0[k])
                     for k in range(n_latents)]
    var_cov0_chol_vecs = \
        svGPFA.utils.miscUtils.getVectorRepOfLowerTrianMatrices(
            lt_matrices=var_cov0_chol)

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

    return initialParams, quadParams, kernels_types


def getIntParam(section_name, param_name, args, config, default_value):
    # command line
    if param_name in vars(args):
        param = int(var(args)[param_name])
    # config
    elif config is not None and \
            section_name in config.sections() and \
            param_name in dict(config.items(section_name)).keys():
        param = int(config[section_name][param_name])
    # default
    else:
        param = default_value
    return param


def getFloatParam(section_name, param_name, args, config, default_value):
    # command line
    if param_name in vars(args):
        param = float(var(args)[param_name])
    # config
    elif config is not None and \
            section_name in config.sections() and \
            param_name in dict(config.items(section_name)).keys():
        param = float(config[section_name][param_name])
    # default
    else:
        param = default_value
    return param


def getStringParam(section_name, param_name, args, config, default_value):
    # command line
    if param_name in vars(args):
        param = var(args)[param_name]
    # config
    elif config is not None and \
            section_name in config.sections() and \
            param_name in dict(config.items(section_name)).keys():
        param = config[section_name][param_name]
    # default
    else:
        param = default_value
    return param


def getLinearEmbeddingParams0(n_neurons, n_latents, args, config=None,
                              embedding_matrix_distribution_dft="Normal",
                              embedding_matrix_loc_dft=0.0,
                              embedding_matrix_scale_dft=1.0,
                              embedding_offset_distribution_dft="Normal",
                              embedding_offset_loc_dft=0.0,
                              embedding_offset_scale_dft=1.0):
    C = getLinearEmbeddingParam0(param_label="c", n_rows=n_neurons,
                           n_cols=n_latents, args=args, config=config,
                           embedding_param_distribution_dft=embedding_matrix_distribution_dft,
                           embedding_param_loc_dft=embedding_matrix_loc_dft,
                           embedding_param_scale_dft=embedding_matrix_scale_dft)
    d = getLinearEmbeddingParam0(param_label="d", n_rows=n_neurons, n_cols=1,
                           args=args, config=config,
                           embedding_param_distribution_dft=embedding_offset_distribution_dft,
                           embedding_param_loc_dft=embedding_offset_loc_dft,
                           embedding_param_scale_dft=embedding_offset_scale_dft)
    C = C.contiguous()
    d = d.contiguous()
    return C, d


def getLinearEmbeddingParam0(param_label, n_rows, n_cols, args, config=None,
                             embedding_param_distribution_dft="Normal",
                             embedding_param_loc_dft=0.0,
                             embedding_param_scale_dft=1.0):
    # Look for param_filename in command line or config
    param_filename = None
    if f"{param_label}_filename" in vars(args).keys() and \
            len(vars(args)[f"{param_label}_filename"]) > 0:
        param_filename = vars(args)[f"{param_label}_filename"]
    elif config is not None and \
            "embedding_params" in config.sections() and \
            f"{param_label}_filename" in dict(config.items("embedding_params")).keys():
        param_filename = config["embedding_params"][f"{param_label}_filename"]

    # If param_filename was found either in the args or in the ini file read
    # param from this filename
    if param_filename is not None:
        df = pd.read_csv(param_filename, header=None)
        param = torch.from_numpy(df.values).type(torch.double)
    else:
        # Else look for param_distribution in the args or in the ini file or
        # use its default function value
        if f"{param_label}_distribution" in vars(args).keys() and \
                len(vars(args)[f"{param_label}_distribution"]) > 0:
            param_distribution = vars(args)[f"{param_label}_distribution"]
            param_loc = float(vars(args)[f"{param_label}_loc"])
            param_scale = float(vars(args)[f"{param_label}_scale"])
        elif config is not None and \
                "embedding_params" in config.sections() and \
                f"{param_label}_distribution" in dict(config.items("embedding_params")).keys():
            param_distribution = config["embedding_params"][f"{param_label}_distribution"]
            param_loc = float(config["embedding_params"][f"{param_label}_loc"])
            param_scale = float(config["embedding_params"][f"{param_label}_scale"])
        else:
            param_distribution = embedding_param_distribution_dft
            param_loc = embedding_param_loc_dft
            param_scale = embedding_param_scale_dft
        if param_distribution == "Normal":
            param = torch.distributions.normal.Normal(
                param_loc, param_scale).sample(sample_shape=[n_rows, n_cols]).type(torch.double)
        else:
            raise ValueError(f"Invalid param_distribution={param_distribution}")

    return param


def getTrialsStartEndTimes(n_trials, args, config,
                           trials_start_time_dft=-1.0, trials_end_time_dft=-1.0):
    trials_start_times = getTrialsTimes(
        param_float_label="trials_start_time",
        param_list_label="trials_start_times",
        n_trials=n_trials,
        args=args, config=config,
        trials_time_dft=trials_start_time_dft)
    trials_end_times = getTrialsTimes(
        param_float_label="trials_end_time",
        param_list_label="trials_end_times",
        n_trials=n_trials,
        args=args, config=config,
        trials_time_dft=trials_end_time_dft)
    return trials_start_times, trials_end_times


def getTrialsTimes(param_list_label, param_float_label, n_trials, args, config,
                   trials_time_dft=-1.0, trials_section_name="data_structure_params"):
    '''In priority order, if available, trial times will be extracted from:
        1. args[param_list_label],
        2. args[param_float_label],
        3. config["trials_params"][param_list_label],
        4. config["trials_params"][param_float_label],
        5. trials_time_dft
        '''

    if param_list_label in vars(args) and \
            len(vars(args)[param_list_label]) > 0:
        trials_times_list = buildFloatListFromStringRep(
            stringRep=vars(args)[param_list_label])
    elif param_float_label in vars(args):
        trials_times_list = [float(vars(args)[param_float_label]) for r in range(n_trials)]
    elif config is not None and trials_section_name in config.sections() and \
            param_list_label in dict(config.items(trials_section_name)).keys():
        trials_times_list = buildFloatListFromStringRep(
            stringRep=config[trials_section_name][param_list_label])
    elif config is not None and trials_section_name in config.sections() and \
            param_float_label in dict(config.items(trials_section_name)).keys():
        trials_times_list = [float(config[trials_section_name][param_float_label])
                             for r in range(n_trials)]
    elif trials_time_dft > 0:
        trials_times_list = [trials_time_dft for r in range(n_trials)]
    else:
        raise RuntimeError(f"If {param_list_label} is not provided as an "
                           "argument, and it is not present in the "
                           "configuration file, then the argument "
                           "trials_time must be specified.")
    trials_times = torch.DoubleTensor(trials_times_list)
    return trials_times


def getKernelsParams0AndTypes(n_latents, args, config=None,
                              k_type_dft="exponentialQuadratic",
                              k_params0_dft=torch.DoubleTensor([1.0])):
    # command line k_type
    if "k_type" in vars(args).keys() and \
            len(vars(args)["k_type"]) > 0:
        if vars(args)["k_type"] == "exponentialQuadratic":
            kernels_types = ["exponentialQuadratic" 
                             for k in range(n_latents)]
            if "k_lengthscale" in vars(args).keys() and \
                    len(vars(args)["k_lengthscale"]) > 0:
                lengthscale = float(vars(args)["k_lengthscale"])
                params0 = [torch.DoubleTensor([lengthscale]) for k in range(n_latents)]
            else:
                raise ValueError("If k_type=exponentialQuadratic is specified "
                                 "in the command line, then k_lengthscale "
                                 "should also be specified in the command "
                                 "line.")
    # config k_type
    elif config is not None and "kernels_params" in config.sections() and \
            "k_type" in dict(config.items("kernels_params")).keys():
        if config["kernels_params"]["k_type"] == "exponentialQuadratic":
            kernels_types = ["exponentialQuadratic" 
                             for k in range(n_latents)]
            lengthscale = float(config["kernels_params"]["k_lengthscale"])
            params0 = [torch.DoubleTensor([lengthscale]) for k in range(n_latents)]
        elif config["kernels_params"]["k_type"] == "periodic":
            kernels_types = ["periodic" for k in range(n_latents)]
            lengthscale = float(config["kernels_params"]["k_engthscale"])
            period = float(config["kernels_params"]["k_period"])
            params0 = [torch.DoubleTensor([lengthscale, period])
                       for k in range(n_latents)]
        else:
            raise RuntimeError(
                f"Invalid kTypeLatents={config['kernels_params']['kTypeLatents']}")
    # config k_type_latent_r
    elif config is not None and "kernels_params" in config.sections() and \
            "k_type_latent0" in dict(config.items("kernels_params")).keys():
        kernels_types = []
        params0 = []
        for k in range(n_latents):
            if config["kernels_params"][f"k_type_latent{k}"] == "exponentialQuadratic":
                kernels_types.append("exponentialQuadratic")
                lengthscaleValue = \
                    float(config["kernels_params"][f"k_lengthscale_latent{k}"])
                params0.append(torch.DoubleTensor([lengthscaleValue]))
            elif config["kernels_params"][f"k_type_latent{k}"] == "periodic":
                kernels_types.append("periodic")
                lengthscaleValue = \
                    float(config["kernels_params"][f"k_lengthscale_latent{k}"])
                periodValue = \
                    float(config["kernels_params"][f"k_period_latent{k}"])
                params0.append(torch.DoubleTensor([lengthscaleValue, periodValue]))
            else:
                raise RuntimeError("Invalid kTypeLatent{:d}={:f}".format(
                    k, config['kernels_params']['kTypeLatent{:d}'.format(k)]))
    # default k_type
    else:
        kernels_types = [k_type_dft for k in range(n_latents)]
        params0 = [k_params0_dft for k in range(n_latents)]

    return params0, kernels_types


def getIndPointsLocs0(n_latents, n_trials, args, config=None,
                      n_ind_points=-1,
                      ind_points_locs_layout="",
                      trials_start_times=None,
                      trials_end_times=None):
    # args ind_points_locs_filename
    if "ind_points_locs_filename" in vars(args).keys() and \
            len(vars(args)["ind_points_locs_filename"]) > 0:
        ind_points_locs_filename = vars(args)["ind_points_locs_filename"]
        ind_points_locs0 = getSameAcrossLatentsAndTrialsIndPointsLocs0(
            n_latents=n_latents, n_trials=n_trials,
            ind_points_locs_filename=ind_points_locs_filename)
    # args ind_points_locs_layout
    elif "ind_points_locs_layout" in vars(args).keys() and \
            len(vars(args)["ind_points_locs_layout"]) > 0 and \
            n_ind_points > 0 and  \
            trials_start_times is not None and \
            trials_end_times is not None:
        layout = vars(args)["ind_points_locs_layout"]
        if layout == "equidistant":
            ind_points_locs0 = buildEquidistantIndPointsLocs0(
                n_latents=n_latents, n_trials=n_trials,
                n_ind_points=n_ind_points,
                trials_start_times=trials_start_times,
                trials_end_times=trials_end_times)
        else:
            raise RuntimeError(f"Invalid ind_points_locs_layout={layout}")
    # config ind_points_locs_filename
    elif config is not None and "ind_points_params" in config.sections() and \
            "ind_points_locs_filename" in dict(config.items("ind_points_params")).keys():
        ind_points_locs_filename = config["ind_points_params"]["ind_points_locs_filename"]
        ind_points_locs0 = getSameAcrossLatentsAndTrialsIndPointsLocs0(
            n_latents=n_latents, n_trials=n_trials,
            ind_points_locs_filename=ind_points_locs_filename)
    # config ind_points_locs_latent<k>_trial<r>_filename
    elif config is not None and "ind_points_params" in config.sections() and \
            "ind_points_locs_latent0_trial0_filename" in dict(config.items("ind_points_params")).keys():
        ind_points_locs0 = getDiffAcrossLatentsAndTrialsIndPointsLocs0(
            n_latents=n_latents, n_trials=n_trials, config=config)
   # config ind_points_locs_layout
    elif config is not None and "ind_points_params" in config.sections() and \
            "ind_points_locs_layout" in dict(config.items("ind_points_params")).keys() and \
            n_ind_points > 0 and  \
            trials_start_times is not None and \
            trials_end_times is not None:
        layout = config["ind_points_params"]["ind_points_locs_layout"]
        if layout == "equidistant":
            ind_points_locs0 = buildEquidistantIndPointsLocs0(
                n_latents=n_latents, n_trials=n_trials,
                n_ind_points=n_ind_points,
                trials_start_times=trials_start_times,
                trials_end_times=trials_end_times)
        else:
            raise RuntimeError(f"Invalid ind_points_locs_layout={layout}")
    # default if no ind_points_locs provided in args or config
    else:
        layout = ind_points_locs_layout
        if n_ind_points < 0:
            raise ValueError("If ind_points_locs0 info does not appear in the "
                             "command line arguments or in the configuration "
                             "file, then n_ind_points should be > 0")
        if trials_start_times is None:
            raise ValueError("If ind_points_locs0 info does not appear in the "
                             "command line arguments or the configuration "
                             "file, then trials_start_times should be "
                             "provided")
        if trials_end_times is None:
            raise ValueError("If ind_points_locs0 info does not appear in the "
                             "command line arguments or the configuration "
                             "file, then trials_end_times should be provided")
        if layout == "equidistant":
            ind_points_locs0 = buildEquidistantIndPointsLocs0(
                n_latents=n_latents, n_trials=n_trials,
                n_ind_points=n_ind_points,
                trials_start_times=trials_start_times,
                trials_end_times=trials_end_times)
        else:
            raise RuntimeError(f"Invalid ind_points_locs_layout={layout}")

    return ind_points_locs0


def getSameAcrossLatentsAndTrialsIndPointsLocs0(
        n_latents, n_trials, ind_points_locs_filename):
    Z0 = torch.from_numpy(pd.read_csv(ind_points_locs_filename, header=None).to_numpy()).flatten()
    Z0s = [[] for k in range(n_latents)]
    nIndPointsForLatent = len(Z0)
    for k in range(n_latents):
        Z0s[k] = torch.empty((n_trials, nIndPointsForLatent, 1),
                             dtype=torch.double)
        Z0s[k][:, :, 0] = Z0
    return Z0s


def getDiffAcrossLatentsAndTrialsIndPointsLocs0(
        n_latents, n_trials, config, section_name="ind_points_params",
        item_name_pattern="ind_points_locs_latent{:d}_trial{:d}_filename"):
    Z0 = [[] for k in range(n_latents)]
    for k in range(n_latents):
        ind_points_locs_filename = config[section_name][item_name_pattern.format(k, 0)]
        Z0_k_r0 = torch.from_numpy(pd.read_csv(ind_points_locs_filename, header=None).to_numpy()).flatten()
        nIndPointsForLatent = len(Z0_k_r0)
        Z0[k] = torch.empty((n_trials, nIndPointsForLatent, 1),
                            dtype=torch.double)
        Z0[k][0, :, 0] = Z0_k_r0
        for r in range(1, n_trials):
            ind_points_locs_filename = config[section_name][item_name_pattern.format(k, r)]
            Z0_k_r = torch.from_numpy(pd.read_csv(ind_points_locs_filename, header=None).to_numpy()).flatten()
            Z0[k][r, :, 0] = Z0_k_r
    return Z0


def buildEquidistantIndPointsLocs0(n_latents, n_trials, n_ind_points,
                                   trials_start_times, trials_end_times):
    Z0s = [[] for k in range(n_latents)]
    for k in range(n_latents):
        Z0s[k] = torch.empty((n_trials, n_ind_points, 1), dtype=torch.double)
        for r in range(n_trials):
            Z0 = torch.linspace(trials_start_times[r], trials_end_times[r], n_ind_points)
            Z0s[k][r, :, 0] = Z0
    return Z0s


def getVariationalMean0(n_latents, n_trials, args, config=None,
                        n_ind_points=-1,
                        constant_value_dft = 0.0,
                        section_name="variational_params",
                        common_filename_item_name="variational_means_filename",
                        different_filename_item_name_pattern="variational_mean_latent{:d}_trial{:d}_filename",
                        constant_value_item_name="variational_mean_constant_value",
                       ):
    # args variational_means_filename
    if common_filename_item_name in vars(args):
        variational_mean0_filename =  args[common_filename_item_name]
        a_variational_mean0 = torch.from_numpy(pd.read_csv(variational_mean0_filename, header=None).to_numpy()).flatten()
        variational_mean0 = getSameAcrossLatentsAndTrialsVariationalMean0(
            n_latents=n_latents, n_trials=n_trials,
            a_variational_mean0=a_variational_mean0)
    # args constant_value
    elif constant_value_item_name in vars(args):
        constant_value = args[constant_value_item_name]
        a_variational_mean0 = constant_value * torch.ones(n_ind_points, dtype=torch.double)
        variational_mean0 = getSameAcrossLatentsAndTrialsVariationalMean0(
            n_latents=n_latents, n_trials=n_trials,
            a_variational_mean0=a_variational_mean0)
    # config variational_means_filename
    elif config is not None and section_name in config.sections() and \
            common_filename_item_name in dict(config.items(section_name)).keys():
        variational_mean0_filename = config[section_name][item_name]
        a_variational_mean0 = torch.from_numpy(pd.read_csv(variational_mean0_filename, header=None).to_numpy()).flatten()
        variational_mean0 = getSameAcrossLatentsAndTrialsVariationalMean0(
            n_latents=n_latents, n_trials=n_trials,
            a_variational_mean0=a_variational_mean0)
    # config variational_means_filename latent k trial r
    elif config is not None and section_name in config.sections() and \
            different_filename_item_name_pattern.format(0, 0) in dict(config.items("variational_params")).keys():
        variational_mean0 = getDiffAcrossLatentsAndTrialsVariationalMean0(
            n_latents=n_latents, n_trials=n_trials, config=config,
            section_name=section_name,
            item_name_pattern=different_filename_item_name_pattern)
    # config constant_value
    elif config is not None and section_name in config.sections() and \
            constant_value_item_name in dict(config.items(section_name)):
        constant_value = config[section_name][constant_value_item_name]
        a_variational_mean0 = constant_value * torch.ones(n_ind_points, dtype=torch.double)
        variational_mean0 = getSameAcrossLatentsAndTrialsVariationalMean0(
            n_latents=n_latents, n_trials=n_trials,
            a_variational_mean0=a_variational_mean0)
    # default constant_value_dft
    else:
        constant_value = constant_value_dft
        a_variational_mean0 = constant_value * torch.ones(n_ind_points, dtype=torch.double)
        variational_mean0 = getSameAcrossLatentsAndTrialsVariationalMean0(
            n_latents=n_latents, n_trials=n_trials,
            a_variational_mean0=a_variational_mean0)
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
        n_latents, n_trials, config,
        section_name="variational_params",
        item_name_pattern="variational_mean_latent{:d}_trial{:d}_filename"):

    variational_mean0 = [[] for r in range(n_latents)]
    for k in range(n_latents):
        variational_mean0_filename = config[section_name][item_name_pattern.format(k, 0)]
        variational_mean0_k0 = torch.from_numpy(pd.read_csv(variational_mean0_filename, header=None).to_numpy()).flatten()
        nIndPointsK = len(variational_mean0_k0)
        variational_mean0[k] = torch.empty((n_trials, nIndPointsK, 1), dtype=torch.double)
        variational_mean0[k][0, :, 0] = variational_mean0_k0
        for r in range(1, n_trials):
            variational_mean0_filename = config[section_name][item_name_pattern.format(k, r)]
            variational_mean0_kr = torch.from_numpy(pd.read_csv(variational_mean0_filename, header=None).to_numpy()).flatten()
            variational_mean0[k][r, :, 0] = variational_mean0_kr
    return variational_mean0


def getVariationalCov0(n_latents, n_trials, args, config=None,
                       n_ind_points=-1,
                       diag_value_dft = 1e-2,
                       section_name="variational_params",
                       common_filename_item_name="variational_covs_filename",
                       different_filename_item_name_pattern="variational_cov_latent{:d}_trial{:d}_filename",
                       diag_value_item_name="variational_cov_diag_value"):
    # args variational_covs_filename
    if common_filename_item_name in vars(args):
        variational_cov0_filename =  args[common_filename_item_name]
        a_variational_cov0 = torch.from_numpy(pd.read_csv(variational_cov0_filename, header=None).to_numpy())
        variational_cov0 = getSameAcrossLatentsAndTrialsVariationalCov0(
            n_latents=n_latents, n_trials=n_trials,
            a_variational_cov0=a_variational_cov0)
    # args diag_value
    elif diag_value_item_name in vars(args):
        diag_value = args[diag_value_item_name]
        a_variational_cov0 = diag_value * torch.eye(n_ind_points, dtype=torch.double)
        variational_mean0 = getSameAcrossLatentsAndTrialsVariationalCov0(
            n_latents=n_latents, n_trials=n_trials,
            a_variational_mean0=a_variational_mean0)
    # config variational_covs_filename
    if config is not None and section_name in config.sections() and \
                       common_filename_item_name in dict(config.items(section_name)):
        variational_cov_filename = config[section_name][common_filename_item_name]
        a_variational_cov0 = torch.from_numpy(pd.read_csv(variational_cov_filename, header=None).to_numpy())
        variational_cov0 = getSameAcrossLatentsAndTrialsVariationalCov0(
            n_latents=n_latents, n_trials=n_trials,
            a_variational_cov0=a_variational_cov0)
    # config variational_cov_latent{:d}_trial{:d}_filename
    elif config is not None and section_name in config.sections() and \
            different_filename_item_name_pattern.format(0, 0) in dict(config.items("variational_params")).keys():
        variational_cov0 = getDiffAcrossLatentsAndTrialsVariationalCov0(
            n_latents=n_latents, n_trials=n_trials, config=config)
    # diag_value_dft
    else:
        diag_value = diag_value_dft
        a_variational_cov0 = diag_value * torch.eye(n_ind_points, dtype=torch.double)
        variational_cov0 = getSameAcrossLatentsAndTrialsVariationalCov0(
            n_latents=n_latents, n_trials=n_trials,
            a_variational_cov0=a_variational_cov0)
    return variational_cov0


def getSameAcrossLatentsAndTrialsVariationalCov0(
        n_latents, n_trials,
        a_variational_cov0,
        section_name="variational_params",
        item_name="variational_covs_filename"):
    variational_cov0 = [[] for r in range(n_latents)]
    nIndPoints = a_variational_cov0.shape[0]
    for k in range(n_latents):
        variational_cov0[k] = torch.empty((n_trials, nIndPoints, nIndPoints),
                                          dtype=torch.double)
        variational_cov0[k][:, :, :] = a_variational_cov0
    return variational_cov0


def getDiffAcrossLatentsAndTrialsVariationalCov0(
        n_latents, n_trials, config,
        section_name="variational_params",
        item_name_pattern="variational_cov_latent{:d}_trial{:d}_filename"):
    variational_cov0 = [[] for r in range(n_latents)]
    for k in range(n_latents):
        variational_cov_filename = config[section_name][item_name_pattern.format(k, 0)]
        variational_cov0_k0 = torch.from_numpy(pd.read_csv(variational_cov_filename, header=None).to_numpy())
        nIndPointsK = variational_cov0_k0.shape[0]
        variational_cov0[k] = torch.empty((n_trials, nIndPointsK, nIndPointsK), dtype=torch.double)
        variational_cov0[k][0, :, :] = variational_cov0_k0
        for r in range(1, n_trials):
            variational_cov_filename = config[section_name][item_name_pattern.format(k, r)]
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
