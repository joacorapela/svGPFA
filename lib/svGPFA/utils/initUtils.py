
import torch
import pandas as pd
import svGPFA.stats.kernelsMatricesStore
import svGPFA.utils.miscUtils


def buildFloatListFromStringRep(stringRep):
    float_list = [float(str) for str in stringRep[1:-1].split(", ")]
    return float_list


def getInitialAndQuadParamsAndKernelsTypes(
        n_neurons, n_trials, args,
        config=None, n_latents_dft=2, n_quad_dft=200,
        embedding_matrix_distribution_dft="Normal",
        embedding_matrix_loc_dft=0.0,
        embedding_matrix_scale_dft=1.0,
        embedding_offset_distribution_dft="Normal",
        embedding_offset_loc_dft=0.0,
        embedding_offset_scale_dft=1.0,
        trials_start_time=-1.0,
        trials_end_time=-1.0,
        k_type_dft="exponentialQuadratic",
        k_params0_dft=torch.DoubleTensor([1.0])):

    n_latents = getNumberParam(section_name="other_params",
                               param_name="n_latents",
                               args=args, config=config,
                               default_value=n_latents_dft)

    n_quad = getNumberParam(section_name="other_params",
                            param_name="n_quad",
                            args=args, config=config,
                            default_value=n_quad_dft)

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
        trials_start_time=trials_start_time,
        trials_end_time=trials_end_time)

    legQuadPoints, legQuadWeights = \
        svGPFA.utils.miscUtils.getLegQuadPointsAndWeights(
            nQuad=n_quad, trials_start_times=trials_start_times,
            trials_end_times=trials_end_times)

    kernels_params0, kernels_types = \
        svGPFA.utils.initUtils.getKernelsParams0AndTypes(
            n_latents=n_latents, args=args, config=config,
            k_type_dft=k_type_dft, k_params0_dft=k_params0_dft)

    ind_points_locs0 = svGPFA.utils.initUtils.getIndPointsLocs0(
        n_latents=n_latents, n_trials=n_trials, args=args, config=config)

    import pdb; pdb.set_trace()

    var_mean0 = svGPFA.utils.initUtils.getVariationalMean0(
        n_latents=n_latents, n_trials=n_trials, config=config)
    var_cov0 = svGPFA.utils.initUtils.getVariationalCov0(
        n_latents=n_latents, n_trials=n_trials, config=config)
    var_cov0_chol = [svGPFA.utils.miscUtils.chol3D(var_cov0[k])
                     for k in range(n_latents)]
    var_cov0_chol_vecs = \
        svGPFA.utils.miscUtils.getVectorRepOfLowerTrianMatrices(
            lt_matrices=var_cov0_chol)

    qUParams0 = {"qMu0": var_mean0, "srQSigma0Vecs": var_cov0_chol_vecs}
    kmsParams0 = {"kernelsParams0": kernels_params0,
                  "inducingPointsLocs0": ind_points_locs0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": d0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    return initialParams, quadParams, kernels_types


def getNumberParam(section_name, param_name, args, config, default_value):
    # command line
    if vars(args)[param_name] > 0:
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


def getLinearEmbeddingParams0(n_neurons, n_latents, args, config=None,
                              embedding_matrix_distribution_dft="Normal",
                              embedding_matrix_loc_dft=0.0,
                              embedding_matrix_scale_dft=1.0,
                              embedding_offset_distribution_dft="Normal",
                              embedding_offset_loc_dft=0.0,
                              embedding_offset_scale_dft=1.0):
    C = getLinearEmbeddingParam0(param_label="C", n_neurons=n_neurons,
                           n_latents=n_latents, args=args, config=config,
                           embedding_param_distribution_dft=embedding_matrix_distribution_dft,
                           embedding_param_loc_dft=embedding_matrix_loc_dft,
                           embedding_param_scale_dft=embedding_matrix_scale_dft)
    d = getLinearEmbeddingParam0(param_label="d", n_neurons=n_neurons, n_latents=n_latents,
                           args=args, config=config,
                           embedding_param_distribution_dft=embedding_offset_distribution_dft,
                           embedding_param_loc_dft=embedding_offset_loc_dft,
                           embedding_param_scale_dft=embedding_offset_scale_dft)
    C = C.contiguous()
    d = d.contiguous()
    return C, d


def getLinearEmbeddingParam0(param_label, n_neurons, n_latents, args, config=None,
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
        param = torch.from_numpy(df.values)
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
                param_loc, param_scale).sample(sample_shape=[n_neurons, n_latents])
        else:
            raise ValueError(f"Invalid param_distribution={param_distribution}")

    return param


def getNumberOfQuadraturePoints(args, config, n_quad_dft):
    if vars(args)["n_quad"] > 0:
        n_quad = int(vars(args)["n_quad"])
    elif config is not None and "n_quad" in config.items("other_param"):
        n_quad = int(config["other_param"]["n_quad"])
    else:
        n_quad = n_quad_dft

    return n_quad


def getTrialsStartEndTimes(n_trials, args, config,
                           trials_start_time=-1.0, trials_end_time=-1.0):
    trials_start_times = getTrialsTimes(
        param_float_label="trials_start_time",
        param_list_label="trials_start_times",
        n_trials=n_trials,
        args=args, config=config,
        trials_time=trials_start_time)
    trials_end_times = getTrialsTimes(
        param_float_label="trials_end_time",
        param_list_label="trials_end_times",
        n_trials=n_trials,
        args=args, config=config,
        trials_time=trials_end_time)
    return trials_start_times, trials_end_times


def getTrialsTimes(param_list_label, param_float_label, n_trials, args, config,
                   trials_time=-1.0, trials_section_name="trials_params"):
    '''In priority order, if available, trial times will be extracted from:
        1. args[param_list_label],
        2. args[param_float_label],
        3. config["trials_params"][param_list_label],
        4. config["trials_params"][param_float_label],
        5. trials_time
        '''

    if param_list_label in vars(args).keys() and \
            len(vars(args)[param_list_label]) > 0:
        trials_times_list = buildFloatListFromStringRep(
            string=vars(args)[param_list_label])
    elif vars(args)[param_float_label] > 0:
        trials_times_list = [float(vars(args)[param_float_label]) for r in range(n_trials)]
    elif config is not None and trials_section_name in config.sections() and \
            param_list_label in dict(config.items(trials_section_name)).keys():
        trials_times_list = buildFloatListFromStringRep(
            string=config[trials_section_name])
    elif config is not None and trials_section_name in config.sections() and \
            param_float_label in dict(config.items(trials_section_name)).keys():
        trials_times_list = [float(config[trials_section_name][param_float_label])
                             for r in range(n_trials)]
    elif trials_time > 0:
        trials_times_list = [trials_time for r in range(n_trials)]
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
                      indPointsLocs_layout_dft="equispaced",
                      n_indPoints=-1,
                      trials_start_times=None,
                      trials_end_times=None):
    # args indPointsLocs_filename
    if "indPointsLocs_filename" in vars(args).keys() and \
            len(vars(args)["indPointsLocs_filename"]) > 0:
        Z0 = buildFloatListFromStringRep(vars(args)["indPointsLocs_filename"])
        ind_points_locs0 = getSameAcrossLatentsAndTrialsIndPointsLocs0(
            n_latents=n_latents, n_trials=n_trials, Z0=Z0)
    # args indPointsLocs_layout
    elif "indPointsLocs_layout" in vars(args).keys() and \
            len(vars(args)["indPointsLocs_layout"]) > 0 and \
            n_indPoints > 0 and  \
            trials_start_times is not None and \
            trials_end_times is not None:
        layout = vars(args)["indPointsLocs_layout"]
        if layout == "equidistant":
            ind_points_locs0 = buildEquidistantIndPointsLocs0(
                n_latents=n_latents, n_trials=n_trials,
                n_indPoints=n_indPoints,
                trials_start_times=trials_start_times,
                trials_end_times=trials_end_times)
        else:
            raise RuntimeError(f"Invalid indPointsLocs_layout={layout}")
    # config indPointsLocs_filename
    elif config is not None and "indPoints_params" in config.sections() and \
            "indPointsLocs_filename" in dict(config.items("indPoints_params")).keys():
        Z0 = buildFloatListFromStringRep(config["indPoints_params"]["indPointsLocs_filename"])
        ind_points_locs0 = getSameAcrossLatentsAndTrialsIndPointsLocs0(
            n_latents=n_latents, n_trials=n_trials, Z0=Z0)
    # config indPointsLocs_latent<k>_trial<r>_filename
    elif config is not None and "indPoints_params" in config.sections() and \
            "indPointsLocs_latent0_trial0_filename" in dict(config.items("indPoints_params")).keys():
        ind_points_locs0 = getDiffAcrossLatentsAndTrialsIndPointsLocs0(
            n_latents=n_latents, n_trials=n_trials, confi=config)
    # config indPointsLocs_layout
    elif config is not None and "indPoints_params" in config.sections() and \
            "indPointsLocs_layout" in dict(config.items("indPoints_params")).keys() and \
            n_indPoints > 0 and  \
            trials_start_times is not None and \
            trials_end_times is not None:
        layout = config["indPoints_params"]["indPointsLocs_layout"]
        if layout == "equidistant":
            ind_points_locs0 = buildEquidistantIndPointsLocs0(
                n_latents=n_latents, n_trials=n_trials,
                n_indPoints=n_indPoints,
                trials_start_times=trials_start_times,
                trials_end_times=trials_end_times)
        else:
            raise RuntimeError(f"Invalid indPointsLocs_layout={layout}")
    # default if no indPointsLocs provided in args or config
    else:
        layout = indPointsLocs_layout_dft
        if n_indPoints < 0:
            raise ValueError("If indPointsLocs0 info does not appear in the "
                             "command line arguments or in the configuration "
                             "file, then n_indPoints should be > 0")
        if trials_start_times is None:
            raise ValueError("If indPointsLocs0 info does not appear in the "
                             "command line arguments or the configuration "
                             "file, then trials_start_times should be "
                             "provided")
        if trials_end_times is None:
            raise ValueError("If indPointsLocs0 info does not appear in the "
                             "command line arguments or the configuration "
                             "file, then trials_end_times should be provided")
        if layout == "equidistant":
            ind_points_locs0 = buildEquidistantIndPointsLocs0(
                n_latents=n_latents, n_trials=n_trials,
                n_indPoints=n_indPoints,
                trials_start_times=trials_start_times,
                trials_end_times=trials_end_times)
        else:
            raise RuntimeError(f"Invalid indPointsLocs_layout={layout}")

    return ind_points_locs0


def getSameAcrossLatentsAndTrialsIndPointsLocs0(
        n_latents, n_trials, Z0, section_name="indPoints_params",
        item_name="indPointsLocsLatentsTrials_filename"):
    Z0s = [[] for k in range(n_latents)]
    nIndPointsForLatent = len(Z0)
    for k in range(n_latents):
        Z0s[k] = torch.empty((n_trials, nIndPointsForLatent, 1),
                             dtype=torch.double)
        Z0s[k][:, :, 0] = Z0
    return Z0s


def getDiffAcrossLatentsAndTrialsIndPointsLocs0(
        n_latents, n_trials, config, section_name="indPoints_params",
        item_name_pattern="indPointsLocsLatent{:d}Trial{:d}_filename"):
    Z0 = [[] for k in range(n_latents)]
    for k in range(n_latents):
        Z0_k_r0 = torch.tensor([float(str) for str in config[section_name][item_name_pattern.format(k, 0)][1:-1].split(", ")], dtype=torch.double)
        nIndPointsForLatent = len(Z0_k_r0)
        Z0[k] = torch.empty((n_trials, nIndPointsForLatent, 1),
                            dtype=torch.double)
        Z0[k][0, :, 0] = Z0_k_r0
        for r in range(1, n_trials):
            Z0[k][r, :, 0] = torch.tensor([float(str) for str in config[section_name][item_name_pattern.format(k, r)][1:-1].split(", ")],
                                          dtype=torch.double)
    return Z0


def buildEquidistantIndPointsLocs0(n_latents, n_trials, n_indPoints,
                                   trials_start_times, trials_end_times):
    Z0s = [[] for k in range(n_latents)]
    for k in range(n_latents):
        Z0s[k] = torch.empty((n_trials, n_indPoints, 1), dtype=torch.double)
        for r in range(n_trials):
            Z0 = torch.linspace(trials_start_times[r], trials_end_times[r], n_indPoints)
            Z0s[k][r, :, 0] = Z0
    return Z0s


def getVariationalMean0(n_latents, n_trials, config=None):
    if "variational_params" in config.sections() and \
            "variational_means_filename" in dict(config.items("variational_params")).keys():
        variational_mean0 = getSameAcrossLatentsAndTrialsVariationalMean0(
            n_latents=n_latents, n_trials=n_trials, config=config)
    elif "variational_params" in config.sections() and \
            "variational_mean_latent0_trial0_filename" in dict(config.items("variational_params")).keys():
        variational_mean0 = getDiffAcrossLatentsAndTrialsVariationalMean0(
            n_latents=n_latents, n_trials=n_trials, confi=config)
    else:
        raise ValueError("Either variational_means_filename or "
                         "variational_mean_latent0_trial0_filename must be "
                         "specified in the configuration file")
#         ind_points_locs0 = buildIndPointsLocsFromConfig(
#             n_latents=n_latents, n_trials=n_trials, config=config)
    return variational_mean0


def getSameAcrossLatentsAndTrialsVariationalMean0(
        n_latents, n_trials, config,
        section_name="variational_params",
        item_name="variational_means_filename"):
    variational_mean0 = [[] for r in range(n_latents)]
    variational_mean0_filename = config[section_name][item_name]
    the_variational_mean0 = torch.from_numpy(pd.read_csv(variational_mean0_filename, header=None).to_numpy()).flatten()
    nIndPoints = len(the_variational_mean0)
    for k in range(n_latents):
        variational_mean0[k] = torch.empty((n_trials, nIndPoints, 1), dtype=torch.double)
        variational_mean0[k][:, :, 0] = the_variational_mean0
    return variational_mean0


def getDiffAcrossLatentsAndTrialsVariationalMean0(
        n_latents, n_trials, config,
        section_name="variational_params",
        item_name_pattern="variationalMeanLatent{:d}Trial{:d}_filename"):

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


def getVariationalCov0(n_latents, n_trials, config=None):
    if "variational_params" in config.sections() and \
            "variational_covs_filename" in dict(config.items("variational_params")).keys():
        variational_cov0 = getSameAcrossLatentsAndTrialsVariationalCov0(
            n_latents=n_latents, n_trials=n_trials, config=config)
    elif "variational_params" in config.sections() and \
            "variational_cov_latent0_trial0_filename" in dict(config.items("variational_params")).keys():
        variational_cov0 = getDiffAcrossLatentsAndTrialsVariationalCov0(
            n_latents=n_latents, n_trials=n_trials, confi=config)
    else:
        raise ValueError("Either variational_covs_filename or "
                         "variational_cov_latent0_trial0_filename must be "
                         "specified in the configuration file")
#         ind_points_locs0 = buildIndPointsLocsFromConfig(
#             n_latents=n_latents, n_trials=n_trials, config=config)
    return variational_cov0


def getSameAcrossLatentsAndTrialsVariationalCov0(
        n_latents, n_trials, config,
        section_name="variational_params",
        item_name="variational_covs_filename"):
    variational_cov0 = [[] for r in range(n_latents)]
    variational_cov_filename = config[section_name][item_name]
    the_variational_cov0 = torch.from_numpy(pd.read_csv(variational_cov_filename, header=None).to_numpy())
    nIndPoints = the_variational_cov0.shape[0]
    for k in range(n_latents):
        variational_cov0[k] = torch.empty((n_trials, nIndPoints, nIndPoints),
                                          dtype=torch.double)
        variational_cov0[k][:, :, :] = the_variational_cov0
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
    indPointsMeans = [[] for r in range(n_trials)]
    for r in range(n_trials):
        indPointsMeans[r] = [[] for k in range(n_latents)]
        for k in range(n_latents):
            indPointsMeans[r][k] = torch.rand(nIndPointsPerLatent[k], 1)*(max-min)+min
    return indPointsMeans


def getConstantIndPointsMeans(constantValue, n_trials, n_latents, nIndPointsPerLatent):
    indPointsMeans = [[] for r in range(n_trials)]
    for r in range(n_trials):
        indPointsMeans[r] = [[] for k in range(n_latents)]
        for k in range(n_latents):
            indPointsMeans[r][k] = constantValue*torch.ones(nIndPointsPerLatent[k], 1, dtype=torch.double)
    return indPointsMeans


def getKzzChol0(kernels, kernelsParams0, indPointsLocs0, epsilon):
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS()
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setKernelsParams(kernelsParams=kernelsParams0)
    indPointsLocsKMS.setIndPointsLocs(indPointsLocs=indPointsLocs0)
    indPointsLocsKMS.setEpsilon(epsilon=epsilon)
    indPointsLocsKMS.buildKernelsMatrices()
    KzzChol0 = indPointsLocsKMS.getKzzChol()
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
