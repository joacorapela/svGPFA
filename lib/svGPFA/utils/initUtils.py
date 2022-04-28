
import pdb
import sys
import os
import torch
import pandas as pd
import svGPFA.stats.kernelsMatricesStore
import svGPFA.utils.miscUtils


def getKernelsParams0AndTypes(nLatents, foreceKernelsUnitScale, config=None,
                              kernel_type_dft="exponentialQuadratic",
                              kernel_params_dft=torch.Tensor([1.0])):
    if config is not None and "kernels_params" in config.sections():
        if "ktypelatents" in dict(config.items("kernels_params")).keys():
            if config["kernels_params"]["kTypeLatents"] == "exponentialQuadratic":
                kernels_types = ["exponentialQuadratic" \
                                 for k in range(nLatents)]
                lengthscaleValue = float(config["kernels_params"]["kLengthscaleValueLatents"])
                params0 = [torch.Tensor([lengthscaleValue]) \
                           for k in range(nLatents)]
            elif config["kernels_params"]["kTypeLatents"] == "periodic":
                kernels_types = ["periodic" for k in range(nLatents)]
                lengthscaleValue = float(config["kernels_params"]["kLengthscaleValueLatents"])
                periodValue = float(config["kernels_params"]["kPeriodValueLatents"])
                params0 = [torch.Tensor([lengthscaleValue, periodValue])
                           for k in range(nLatents)]
            else:
                raise RuntimeError(
                    f"Invalid kTypeLatents={config['kernels_params']['kTypeLatents']}")
        elif "ktypelatent0" in dict(config.items("kernels_params")).keys():
            kernels_types = []
            params0 = []
            for k in range(nLatents):
                if config["kernels_params"][f"ktypelatent{k}"] == "exponentialQuadratic":
                    kernels_types.append("exponentialQuadratic")
                    lengthscaleValue = \
                        float(config["kernels_params"][f"klengthscalevaluelatent{k}"])
                    params0.append(torch.Tensor([lengthscaleValue]))
                elif config["kernels_params"][f"kTypeLatent{k}"] == "periodic":
                    kernels_types.append("periodic")
                    lengthscaleValue = \
                        float(config["kernels_params"][f"kLengthscaleValueLatent{k}"])
                    periodValue = \
                        float(config["kernels_params"][f"kPeriodValueLatent{k}"])
                    params0.append(torch.Tensor([lengthscaleValue,
                                                 periodValue]))
                else:
                    raise RuntimeError("Invalid kTypeLatent{:d}={:f}".format(
                        k, config['kernels_params']['kTypeLatent{:d}'.format(k)]))
        else:
            raise RuntimeError("Either item ktypelatents or item ktypelatent0 "
                               "should appear under section kernels_params")
    else:
        kernels_types = [kernel_type_dft for k in range(nLatents)]
        params0 = [torch.Tensor([kernel_params_dft]) for k in range(nLatents)]

    return params0, kernels_types


def getEmbeddingParams0(nNeurons, nLatents, config=None,
                        C_mean_dft=0, C_std_dft=0.01,
                        d_mean_dft=0, d_std_dft=0.01,
                        ):
    if config is not None and \
       "embedding_params" in config.section() and \
       "C_filename" in dict(config.items("embedding_params")).keys() and \
       "d_filename" in dict(config.items("embedding_params")).keys():
        CFilename = config["embedding_params"]["C_filename"]
        dFilename = config["embedding_params"]["d_filename"]
        C0, d0 = svGPFA.utils.configUtils.getLinearEmbeddingParams(
            CFilename=CFilename, dFilename=dFilename)
    else:
        # C default to N(C_mean_dft, C_std_dft)
        # d default to N(d_mean_dft, d_std_dft)
        C0 = torch.normal(C_mean_dft, C_std_dft, size=(nNeurons, nLatents),
                          dtype=torch.double).contiguous()
        d0 = torch.normal(d_mean_dft, d_std_dft, size=(nNeurons, 1),
                          dtype=torch.double).contiguous()
    return C0, d0


def getIndPointsLocs)(nLatents, nTrials, config=None,
                      ind_points_locs0_layout="equispaced"):
    if config is not None and \
       "indPoints_params" in config.section() and \
       "indPointsLocsLatentsTrials" in dict(config.items("indPoints_params")).keys():
        Z0 = getSameAcrossLatentsAndTrialsIndPointsLocs0(nLatents=nLatents,
                                                         nTrials=nTrials,
                                                         config=config)
    elif config is not None and \
            "indPoints_params" in config.section() and \
            "indPointsLocsLatent0Trial0" in dict(config.items("indPoints_params")).keys():
        Z0 = getIndPointsLocs0(nLatents=nLatents, nTrials=nTrials,
                               confi=config)
    else:
        n_ind_points = int(config["indPoints_params"]["nIndPoints"])
        if ind_points_locs0_layout is None:
            ind_points_locs0_layout = \
                config["indPoints_params"]["ind_points_locs0_layout"]
        if "trialsStartTime" in dict(config.items("trials_params")):
            trials_start_time = float(config["trials_params"]["trialsStartTime"])
            trials_start_times = [trialsStartTime for r in range(nTrials)]
        elif "trial0StartTime" in dict(config.items("trials_params")):
            trials_start_times = \
                [float(config[f"trials_params"]["trial{r}StartTime"]) for r in range(nTrials)]
        if "trialsEndTime" in dict(config.items("trials_params")):
            trials_end_time = float(config["trials_params"]["trialsEndTime"])
            trials_end_times = [trialsEndTime for r in range(nTrials)]
        elif "trial0EndTime" in dict(config.items("trials_params")):
            trials_end_times = \
                [float(config[f"trials_params"]["trial{r}EndTime"]) for r in range(nTrials)]
        raise RuntimeError("Either item indPointsLocsLatentsTrials or "
                           "item indPointsLocsLatent0Trial0 should appear in "
                           "section indPoints_params")
    return Z0


def getParams0AndKernelsTypes(nNeurons, nLatents, config=None,
                              C_mean_dft=0, C_std_dft=0.01,
                              d_mean_dft=0, d_std_dft=0.01,
                              forceKernelsUnitScale=True,
                              ):
    kernelsScaledParams0, kernelsTypes = getKernelsParams0AndTypes(
        nLatents=nLatents, forceKernelsUnitScale=forceKernelsUnitScale,
        config=config)
    C0, d0 = getEmbeddingParams0(nNeurons=nNeurons, nLatents=nLatents,
                                 config=config,
                                 C_mean_dft=C_mean_dft, C_std_dft=C_std_dft,
                                 d_mean_dft=d_mean_dft, d_std_dft=d_std_dft)
    qMu0 = getIndPointsParams(prefix="indPointsLoc", config=config)

def getUniformIndPointsMeans(nTrials, nLatents, nIndPointsPerLatent, min=-1, max=1):
    indPointsMeans = [[] for r in range(nTrials)]
    for r in range(nTrials):
        indPointsMeans[r] = [[] for k in range(nLatents)]
        for k in range(nLatents):
            indPointsMeans[r][k] = torch.rand(nIndPointsPerLatent[k], 1)*(max-min)+min
    return indPointsMeans

def getConstantIndPointsMeans(constantValue, nTrials, nLatents, nIndPointsPerLatent):
    indPointsMeans = [[] for r in range(nTrials)]
    for r in range(nTrials):
        indPointsMeans[r] = [[] for k in range(nLatents)]
        for k in range(nLatents):
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

def getScaledIdentityQSigma0(scale, nTrials, nIndPointsPerLatent):
    nLatent = len(nIndPointsPerLatent)
    qSigma0 = [[None] for k in range(nLatent)]

    for k in range(nLatent):
        qSigma0[k] = torch.empty((nTrials, nIndPointsPerLatent[k], nIndPointsPerLatent[k]), dtype=torch.double)
        for r in range(nTrials):
            qSigma0[k][r,:,:] = scale*torch.eye(nIndPointsPerLatent[k], dtype=torch.double)
    return qSigma0

def getSVPosteriorOnIndPointsParams0(nIndPointsPerLatent, nLatents, nTrials, scale):
    qMu0 = [[] for k in range(nLatents)]
    qSVec0 = [[] for k in range(nLatents)]
    qSDiag0 = [[] for k in range(nLatents)]
    for k in range(nLatents):
        # qMu0[k] = torch.rand(nTrials, nIndPointsPerLatent[k], 1, dtype=torch.double)
        qMu0[k] = torch.zeros(nTrials, nIndPointsPerLatent[k], 1, dtype=torch.double)
        qSVec0[k] = scale*torch.eye(nIndPointsPerLatent[k], 1, dtype=torch.double).repeat(nTrials, 1, 1)
        qSDiag0[k] = scale*torch.ones(nIndPointsPerLatent[k], 1, dtype=torch.double).repeat(nTrials, 1, 1)
    return qMu0, qSVec0, qSDiag0

def getKernels(nLatents, config, forceUnitScale):
    kernels = [[] for r in range(nLatents)]
    for k in range(nLatents):
        kernelType = config["kernel_params"]["kTypeLatent{:d}".format(k)]
        if kernelType=="periodic":
            if not forceUnitScale:
                scale = float(config["kernel_params"]["kScaleValueLatent{:d}".format(k)])
            else:
                scale = 1.0
            lengthscale = float(config["kernel_params"]["kLengthscaleScaledValueLatent{:d}".format(k)])
            period = float(config["kernel_params"]["kPeriodScaledValueLatent{:d}".format(k)])
            kernel = svGPFA.stats.kernels.PeriodicKernel(scale=scale)
            kernel.setParams(params=torch.Tensor([lengthscale, period]).double())
        elif kernelType=="exponentialQuadratic":
            if not forceUnitScale:
                scale = float(config["kernel_params"]["kScaleValueLatent{:d}".format(k)])
            else:
                scale = 1.0
            lengthscale = float(config["kernel_params"]["kLengthscaleScaledValueLatent{:d}".format(k)])
            kernel = svGPFA.stats.kernels.ExponentialQuadraticKernel(scale=scale)
            kernel.setParams(params=torch.Tensor([lengthscale]).double())
        else:
            raise ValueError("Invalid kernel type {:s} for latent {:d}".format(kernelType, k))
        kernels[k] = kernel
    return kernels


def getKernelsParams0(kernels, noiseSTD):
    nLatents = len(kernels)
    kernelsParams0 = [kernels[k].getParams() for k in range(nLatents)]
    if noiseSTD > 0.0:
        kernelsParams0 = [kernelsParams0[0] +
                          noiseSTD*torch.randn(len(kernelsParams0[k]))
                          for k in range(nLatents)]
    return kernelsParams0

def getKernelsScaledParams0(kernels, noiseSTD):
    nLatents = len(kernels)
    kernelsParams0 = [kernels[k].getScaledParams() for k in range(nLatents)]
    if noiseSTD > 0.0:
        kernelsParams0 = [kernelsParams0[0] +
                          noiseSTD*torch.randn(len(kernelsParams0[k]))
                          for k in range(nLatents)]
    return kernelsParams0

def getSRQSigmaVecsFromKzz(Kzz):
    Kzz_chol = []
    for aKzz in Kzz:
        Kzz_chol.append(svGPFA.utils.miscUtils.chol3D(aKzz))
    answer = getSRQSigmaVecsFromSRMatrices(srMatrices=Kzz_chol)
    return answer


def getSRQSigmaVecsFromSRMatrices(srMatrices):
    """Returns vectors containing the lower-triangular elements of the input
    lower- triangular matrices.

    :param srMatrices: a list of length nLatents, with srMatrices[k] a tensor
    of dimension nTrials x nIndPoints x nIndPoints, where
    srMatrices[k][r, :, :] is a lower-triangular matrix.

    :type srMatrices: list
    :return: a list srQSigmaVec of length nLatents, whith srQSigmaVec[k] a
    tensor of dimension nTrials x (nIndPoints+1)*nIndPoints/2 x 0, where
    srQSigmaVec[k][r, :, 0] contains the lower-triangular elements of
    srMatrices[k][r, :, :]
    """

    nLatents = len(srMatrices)
    nTrials = srMatrices[0].shape[0]

    srQSigmaVec = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        nIndPointsK = srMatrices[k].shape[1]
        Pk = int((nIndPointsK+1)*nIndPointsK/2)
        srQSigmaVec[k] = torch.empty((nTrials, Pk, 1), dtype=torch.double)
        for r in range(nTrials):
            cholKR = srMatrices[k][r,:,:]
            trilIndices = torch.tril_indices(nIndPointsK, nIndPointsK)
            cholKRVec = cholKR[trilIndices[0,:], trilIndices[1,:]]
            srQSigmaVec[k][r,:,0] = cholKRVec
    return srQSigmaVec

def getIndPointsLocs0(nLatents, nTrials, config):
    Z0 = [[] for k in range(nLatents)]
    for k in range(nLatents):
        option_array = "indPointsLocsLatent{:d}Trial{:d}".format(k,0).lower()
        option_filename = "indPointsLocsLatent{:d}Trial{:d}_filename".format(k,0).lower()
        if option_array in config.options("indPoints_params"):
            Z0_k_r0 = torch.tensor([float(str) for str in config["indPoints_params"]["indPointsLocsLatent{:d}Trial{:d}".format(k,0)][1:-1].split(", ")], dtype=torch.double)
        elif option_filename in config.options("indPoints_params"):
            Z0_k_r0 = torch.from_numpy(np.loadtxt(config["indPoints_params"]["indPointsLocsLatent{:d}Trial{:d}_filename".format(k,0)], delimiter=","))
        else:
            raise ValueError("option={:s} not found in config.options('indPoints_params')")
        nIndPointsForLatent = len(Z0_k_r0)
        Z0[k] = torch.empty((nTrials, nIndPointsForLatent, 1), dtype=torch.double)
        Z0[k][0,:,0] = Z0_k_r0
        for r in range(1, nTrials):
            option_array = "indPointsLocsLatent{:d}Trial{:d}".format(k,r).lower()
            option_filename = "indPointsLocsLatent{:d}Trial{:d}_filename".format(k,r).lower()
            if option_array in config.options("indPoints_params"):
                Z0[k][r,:,0] = torch.tensor([float(str) for str in config["indPoints_params"]["indPointsLocsLatent{:d}Trial{:d}".format(k,r)][1:-1].split(", ")], dtype=torch.double)
            elif option_filename in config.options("indPoints_params"):
                Z0[k][r,:,0] = torch.from_numpy(np.loadtxt(config["indPoints_params"]["indPointsLocsLatent{:d}Trial{:d}_filename".format(k,r)], delimiter=","))
            else:
                raise ValueError("option={:s} not found in config.options('indPoints_params')")
    return Z0


def getIdenticalIndPointsLocs0(nLatents, nTrials, config):
    Z0 = [[] for k in range(nLatents)]
    for k in range(nLatents):
        the_Z0 = torch.tensor([float(str) for str in config["indPoints_params"]["indPointsLocs"][1:-1].split(", ")],
                              dtype=torch.double)
        nIndPointsForLatent = len(the_Z0)
        Z0[k] = torch.empty((nTrials, nIndPointsForLatent, 1),
                            dtype=torch.double)
        Z0[k][:, :, 0] = the_Z0
    return Z0


def getVariationalMean0(nLatents, nTrials, config, keyNamePattern="qMu0Latent{:d}Trial{:d}_filename"):
    qMu0 = [[] for r in range(nLatents)]
    for k in range(nLatents):
        qMu0Filename = config["variational_params"][keyNamePattern.format(k, 0)]
        qMu0k0 = torch.from_numpy(pd.read_csv(qMu0Filename, header=None).to_numpy()).flatten()
        nIndPointsK = len(qMu0k0)
        qMu0[k] = torch.empty((nTrials, nIndPointsK, 1), dtype=torch.double)
        qMu0[k][0,:,0] = qMu0k0
        for r in range(1, nTrials):
            qMu0Filename = config["variational_params"][keyNamePattern.format(k, r)]
            qMu0kr = torch.from_numpy(pd.read_csv(qMu0Filename, header=None).to_numpy()).flatten()
            qMu0[k][r,:,0] = qMu0kr
    return qMu0


def getIdenticalVariationalMean0(nLatents, nTrials, config,
                                 keyName="qMu0_filename"):
    qMu0 = [[] for r in range(nLatents)]
    qMu0Filename = config["variational_params"][keyName]
    the_qMu0 = torch.from_numpy(pd.read_csv(qMu0Filename, header=None).to_numpy()).flatten()
    nIndPoints = len(the_qMu0)
    for k in range(nLatents):
        qMu0[k] = torch.empty((nTrials, nIndPoints, 1), dtype=torch.double)
        qMu0[k][:, :, 0] = the_qMu0
    return qMu0


def getVariationalCov0(nLatents, nTrials, config, keyNamePattern="qSigma0Latent{:d}Trial{:d}_filename"):
    qSigma0 = [[] for r in range(nLatents)]
    for k in range(nLatents):
        qSigma0Filename = config["variational_params"][keyNamePattern.format(k, 0)]
        qSigma0k0 = torch.from_numpy(pd.read_csv(qSigma0Filename, header=None).to_numpy())
        nIndPointsK = qSigma0k0.shape[0]
        qSigma0[k] = torch.empty((nTrials, nIndPointsK, nIndPointsK), dtype=torch.double)
        qSigma0[k][0,:,:] = qSigma0k0
        for r in range(1, nTrials):
            qSigma0Filename = config["variational_params"][keyNamePattern.format(k, r)]
            qSigma0kr = torch.from_numpy(pd.read_csv(qSigma0Filename, header=None).values)
            qSigma0[k][r,:,:] = qSigma0kr
    return qSigma0


def getIdenticalVariationalCov0(nLatents, nTrials, config,
                                keyName="qSigma0_filename"):
    qSigma0 = [[] for r in range(nLatents)]
    qSigma0Filename = config["variational_params"][keyName]
    the_qSigma0 = torch.from_numpy(pd.read_csv(qSigma0Filename, header=None).to_numpy())
    nIndPoints = the_qSigma0.shape[0]
    for k in range(nLatents):
        qSigma0[k] = torch.empty((nTrials, nIndPoints, nIndPoints),
                                 dtype=torch.double)
        # qSigma0[k][r,:,:] = qSigma0kr
        qSigma0[k][:, :, :] = the_qSigma0
    return qSigma0


