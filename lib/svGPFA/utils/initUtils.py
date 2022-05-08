
import pdb
import sys
import os
import torch
import pandas as pd
import svGPFA.stats.kernelsMatricesStore
import svGPFA.utils.miscUtils


def getKernelsParams0AndTypes(n_latents, config=None, foreceKernelsUnitScale=True):
    if "k_type" in dict(config.items("kernels_params")).keys():
        if config["kernels_params"]["k_type"] == "exponentialQuadratic":
            kernels_types = ["exponentialQuadratic" \
                             for k in range(n_latents)]
            lengthscale = float(config["kernels_params"]["k_lengthscale"])
            params0 = [torch.Tensor([lengthscale]) for k in range(n_latents)]
        elif config["kernels_params"]["k_type"] == "periodic":
            kernels_types = ["periodic" for k in range(n_latents)]
            lengthscale = float(config["kernels_params"]["k_engthscale"])
            period = float(config["kernels_params"]["k_period"])
            params0 = [torch.Tensor([lengthscale, period])
                       for k in range(n_latents)]
        else:
            raise RuntimeError(
                f"Invalid kTypeLatents={config['kernels_params']['kTypeLatents']}")
    elif "k_type_latent0" in dict(config.items("kernels_params")).keys():
        kernels_types = []
        params0 = []
        for k in range(n_latents):
            if config["kernels_params"][f"k_type_latent{k}"] == "exponentialQuadratic":
                kernels_types.append("exponentialQuadratic")
                lengthscaleValue = \
                    float(config["kernels_params"][f"k_lengthscale_latent{k}"])
                params0.append(torch.Tensor([lengthscaleValue]))
            elif config["kernels_params"][f"k_type_latent{k}"] == "periodic":
                kernels_types.append("periodic")
                lengthscaleValue = \
                    float(config["kernels_params"][f"k_lengthscale_latent{k}"])
                periodValue = \
                    float(config["kernels_params"][f"k_period_latent{k}"])
                params0.append(torch.Tensor([lengthscaleValue, periodValue]))
            else:
                raise RuntimeError("Invalid kTypeLatent{:d}={:f}".format(
                    k, config['kernels_params']['kTypeLatent{:d}'.format(k)]))
    else:
        raise RuntimeError("Either item ktypelatents or item ktypelatent0 "
                           "should appear under section kernels_params")

    return params0, kernels_types


def getLinearEmbeddingParams0(n_neurons, n_latents, args, config=None,
                              embedding_matrix_distribution_dft="Normal",
                              embedding_matrix_loc_dft=0.0,
                              embedding_matrix_scale_dft=1.0,
                              embedding_offset_distribution_dft="Normal",
                              embedding_offset_loc_dft=0.0,
                              embedding_offset_scale_dft=1.0):
    C = getEmbeddingParam0(param_label="C", n_neurons=n_neurons, n_latents=n_latents,
                           args=args, config=config,
                           embedding_param_distribution_dft= embedding_matrix_distribution_dft,
                           embedding_param_loc_dft=embedding_matrix_loc_dft,
                           embedding_param_scale_dft=embedding_matrix_scale_dft)
    d = getEmbeddingParam0(param_label="d", n_neurons=n_neurons, n_latents=n_latents,
                           args=args, config=config,
                           embedding_param_distribution_dft= embedding_offset_distribution_dft,
                           embedding_param_loc_dft=embedding_offset_loc_dft,
                           embedding_param_scale_dft=embedding_offset_scale_dft)
    return C, d



def getEmbeddingParam0(param_label, n_neurons, n_latents, args, config=None,
                       embedding_param_distribution_dft="Normal",
                       embedding_param_loc_dft=0.0,
                       embedding_param_scale_dft=1.0)
    param_filename = None
    if len(args[f"{param_label}_filename"])>0:
        param_filename = args[param_label]
    elif config is not None and \
            "embedding_params" in config.sections() \
            param_label in dict(config.items("embedding_params")).keys():
        param_Filename = config["embedding_params"][f"{param_label}_filename"]

    # If param_filename was found either in the args or in the ini file read
    # param from this filename
    if param_filename is not None:
        df = pd.read_csv(param_Filename, header=None)
        param = torch.from_numpy(df.values)
    else:
        # Else look for param_distribution in the args or in the ini file or use
        # its default function value
        if len(args[f"{param_label}_distribution"])>0:
            param_distribution = args[f"{param_label}_distribution"]
            param_loc = float(args[f"{param_label}_loc"])
            param_scale = float(args[f"{param_label}_scale"])
        elif config is not None and \
                "embedding_params" in config.sections() \
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


def getIndPointsLocs0(n_latents, n_trials, config=None):
    if "indPointsLocsLatentsTrials_filename" in dict(config.items("indPoints_params")).keys():
        ind_points_locs0 = getSameAcrossLatentsAndTrialsIndPointsLocs0(
            n_latents=n_latents, n_trials=n_trials, config=config)
    elif "indPointsLocsLatent0Trial0_filename" in dict(config.items("indPoints_params")).keys():
        ind_points_locs0 = getDiffAcrossLatentsAndTrialsIndPointsLocs0(
            n_latents=n_latents, n_trials=n_trials, confi=config)
    else:
        ind_points_locs0 = buildIndPointsLocsFromConfig(
            n_latents=n_latents, n_trials=n_trials, config=config)
    return ind_points_locs0


def getSameAcrossLatentsAndTrialsIndPointsLocs0(
        n_latents, n_trials, config, section_name="indPoints_params",
        item_name="indPointsLocsLatentsTrials_filename"):
    Z0 = [[] for k in range(n_latents)]
    for k in range(n_latents):
        the_Z0 = torch.tensor([float(str) for str in config[section_name][item_name][1:-1].split(", ")], dtype=torch.double)
        nIndPointsForLatent = len(the_Z0)
        Z0[k] = torch.empty((n_trials, nIndPointsForLatent, 1),
                            dtype=torch.double)
        Z0[k][:, :, 0] = the_Z0
    return Z0


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


def buildIndPointsLocsFromConfig(n_latents, n_trials, config):
    if "n_ind_points" in dict(config.items("indPoints_params")).keys():
        n_ind_points = int(config["indPoints_params"]["n_ind_points"])
    else:
        raise ValueError("n_ind_points should appear in section "
                         "indPoints_params")
    if "ind_points_locs0_layout" in dict(config.items("indPoints_params")).keys():
        ind_points_locs0_layout = config["indPoints_params"]["ind_points_locs0_layout"]
    else:
        raise ValueError("ind_points_locs0_layout should appear in section "
                         "indPoints_params")

    trials_start_times, trials_end_times = getTrialsStartEndTimes(
        n_trials=n_trials, config=config)

    ind_points_locs0 = buildIndPointsLocs0(n_latents=n_latents,
                                           n_trials=n_trials,
                                           n_ind_points=n_ind_points,
                                           layout=ind_points_locs0_layout,
                                           trials_start_time=trials_start_times[0],
                                           trials_end_time=trials_end_times[0])
    return ind_points_locs0


def getTrialsStartEndTimes(n_trials, config):
    if "trials_start_time" in dict(config.items("trials_params")):
        trials_start_time = float(config["trials_params"]["trials_start_time"])
        trials_start_times = [trials_start_time for r in range(n_trials)]
    elif "trial0_start_time" in dict(config.items("trials_params")):
        trials_start_times = \
            [float(config["trials_params"]["trial{r}_start_time"])
             for r in range(n_trials)]
    else:
        raise ValueError("Items trialsStartTime or trial0StartTime are "
                         "missing in section trials_params")
    if "trials_end_time" in dict(config.items("trials_params")):
        trials_end_time = float(config["trials_params"]["trials_end_time"])
        trials_end_times = [trials_end_time for r in range(n_trials)]
    elif "trial0_end_time" in dict(config.items("trials_params")):
        trials_end_times = \
            [float(config[f"trials_params"]["trial{r}_end_time"]) for r in range(n_trials)]
    else:
        raise ValueError("Items trialsEndTime or trial0EndTime are missing "
                         "in section trials_params")
    return trials_start_times, trials_end_times


def buildIndPointsLocs0(n_latents, n_trials, n_ind_points, layout,
                        trials_start_time, trials_end_time):
    the_Z0 = torch.linspace(trials_start_time, trials_end_time, n_ind_points)
    Z0 = [[] for k in range(n_latents)]
    for k in range(n_latents):
        Z0[k] = torch.empty((n_trials, n_ind_points, 1), dtype=torch.double)
        Z0[k][:, :, 0] = the_Z0
    return Z0


def getVariationalMean0(n_latents, n_trials, config=None):
    if "variational_means_filename" in dict(config.items("variational_params")).keys():
        variational_mean0 = getSameAcrossLatentsAndTrialsVariationalMean0(
            n_latents=n_latents, n_trials=n_trials, config=config)
    elif "variational_mean_latent0_trial0_filename" in dict(config.items("variational_params")).keys():
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
    if "variational_covs_filename" in dict(config.items("variational_params")).keys():
        variational_cov0 = getSameAcrossLatentsAndTrialsVariationalCov0(
            n_latents=n_latents, n_trials=n_trials, config=config)
    elif "variational_cov_latent0_trial0_filename" in dict(config.items("variational_params")).keys():
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


def getParams0AndKernelsTypes(nNeurons, n_latents, config=None,
                              C_mean_dft=0, C_std_dft=0.01,
                              d_mean_dft=0, d_std_dft=0.01,
                              forceKernelsUnitScale=True,
                              ):
    kernelsScaledParams0, kernelsTypes = getKernelsParams0AndTypes(
        n_latents=n_latents, forceKernelsUnitScale=forceKernelsUnitScale,
        config=config)
    C0, d0 = getEmbeddingParams0(nNeurons=nNeurons, n_latents=n_latents,
                                 config=config,
                                 C_mean_dft=C_mean_dft, C_std_dft=C_std_dft,
                                 d_mean_dft=d_mean_dft, d_std_dft=d_std_dft)
    qMu0 = getIndPointsParams(prefix="indPointsLoc", config=config)


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

def getSRQSigmaVecsFromKzz(Kzz):
    Kzz_chol = []
    for aKzz in Kzz:
        Kzz_chol.append(svGPFA.utils.miscUtils.chol3D(aKzz))
    answer = getSRQSigmaVecsFromSRMatrices(srMatrices=Kzz_chol)
    return answer


def getInitialAndQuadParamsAndKernelsTypes(n_neurons, n_trials,
                                           args, config=None,
                                           n_latents_dft=2,
                                           n_quad_dft=200,
                                           embedding_matrix_distribution_dft="Normal",
                                           embedding_matrix_loc_dft=0.0,
                                           embedding_matrix_scale_dft=1.0,
                                           embedding_offset_distribution_dft="Normal",
                                           embedding_offset_loc_dft=0.0,
                                           embedding_offset_scale_dft=1.0,
                                           ):
    if args["n_latents"]>0:
        n_latents = int(args["n_latents"])
    elif config is not None and \
            "other_param" in config.sections() and \
            "n_latents" in dict(config.items("other_param")).keys():
        n_latents = int(config["other_param"]["n_latents"])
    else:
        n_latents = n_latents_dft

    C0, d0 = svGPFA.utils.initUtils.getLinearEmbeddingParams0(
        args=args, config=config,
        embedding_matrix_distribution_dft=embedding_matrix_distribution_dft,
        embedding_matrix_loc_dft=embedding_matrix_loc_dft,
        embedding_matrix_scale_dft=embedding_matrix_scale_dft,
        embedding_offset_distribution_dft=embedding_offset_distribution_dft,
        embedding_offset_loc_dft=embedding_offset_loc_dft,
        embedding_offset_scale_dft=embedding_offset_scale_dft,
    )
    C0 = C0.contiguous()
    d0 = d0.contiguous()

    if args["n_quad"]>0:
        n_quad = int(args["n_quad"])
    elif config is not None and "n_quad" in config.items("other_param"):
        n_latents = int(config["other_param"]["n_quad"])
    else:
        n_quad = n_quad_dft

    trials_start_times, trials_end_times = getTrialsStartEndTimes(
        n_trials=n_trials, config=config)

    legQuadPoints, legQuadWeights = \
        svGPFA.utils.miscUtils.getLegQuadPointsAndWeights(
            nQuad=n_quad, trials_start_times=trials_start_times,
            trials_end_times=trials_end_times)

    kernels_params0, kernels_types = \
        svGPFA.utils.initUtils.getKernelsParams0AndTypes(n_latents=n_latents,
                                                         config=config)
    ind_points_locs0 = svGPFA.utils.initUtils.getIndPointsLocs0(
        n_latents=n_latents, n_trials=n_trials, config=config)
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
