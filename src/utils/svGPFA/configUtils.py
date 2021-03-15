
import pdb
import math
import pandas as pd
import torch
import stats.kernels

def getParamsLogPriors(nLatents, config):
    embeddingLogPriors = _getEmbeddingLogPriors(config=config)
    kernelsLogPriors = _getKernelsLogPriors(nLatents=nLatents, config=config)
    indPointsLocsLogPriors = _getIndPointsLocsLogPriors(config=config)
    answer = {
        "embedding": embeddingLogPriors,
        "kernels": kernelsLogPriors,
        "indPointsLocs": indPointsLocsLogPriors
    }
    return answer

def _getIndPointsLocsLogPriors(config):
    def indPointLocsLogPrior(indPointsLocs, indPointsLocs_element_log_prior):
        nLatents = len(indPointsLocs)
        indPointsLocsLogProb = 0.0
        for k in range(nLatents):
            nTrials = indPointsLocs[k].shape[2]
            for r in range(nTrials):
                for indPointLoc in indPointsLocs[k][0,:,r]:
                    indPointsLocsLogProb = indPointsLocsLogProb + indPointsLocs_element_log_prior(indPointLoc)
        # pdb.set_trace()
        return indPointsLocsLogProb

    indPointsLocs_prior_type = config["indPoints_params"]["indPointsLocs_prior_type"]
    if indPointsLocs_prior_type == "uniform":
        indPointsLocs_prior_min = float(config["indPoints_params"]["indPointsLocs_prior_min"])
        indPointsLocs_prior_max = float(config["indPoints_params"]["indPointsLocs_prior_max"])
        def indPointsLocs_element_log_prior(x, min=indPointsLocs_prior_min, max=indPointsLocs_prior_max):
            uniform = torch.distributions.uniform.Uniform(min, max)
            answer = uniform.log_prob(x)
            return answer
    else:
        raise NotImplementedError("Prior for type {:s} for the inducing points locations has not been implemented yet".format(C_prior_type))

    answer = lambda indPointsLocs: indPointLocsLogPrior(indPointsLocs=indPointsLocs, indPointsLocs_element_log_prior=indPointsLocs_element_log_prior)
    return answer


def _getKernelsLogPriors(nLatents, config):
    kernelsLogPriors = [None]*nLatents
    for i in range(nLatents):
        kernelsLogPriors[i] = _getKernelLogPrior(latent=i, config=config)
    return kernelsLogPriors

def _getKernelLogPrior(latent, config):
    def kernelParamsLogPrior(kernelParams, kernelScales, lengthscaleLogPrior, periodLogPrior):
        lengthscale = kernelParams[0]/kernelScales[0]
        lengthscaleLogProb = lengthscaleLogPrior(lengthscale)

        period = kernelParams[1]/kernelScales[1]
        periodLogProb = periodLogPrior(period)

        answer = lengthscaleLogProb + periodLogProb
        # pdb.set_trace()
        return answer

    kernel_type = config["kernel_params"]["kTypeLatent{:d}".format(latent)]
    if kernel_type == "periodic":
        lengthscaleScale = float(config["kernel_params"]["kLengthscaleScaleLatent{:d}".format(latent)])
        periodScale = float(config["kernel_params"]["kPeriodScaleLatent{:d}".format(latent)])
        kernelScales = [lengthscaleScale, periodScale]

        lengthscale_prior_type = config["kernel_params"]["kLengthscaleLatent{:d}_prior_type".format(latent)]
        if lengthscale_prior_type=="normal":
            lengthscale_prior_mean = float(config["kernel_params"]["kLengthscaleLatent{:d}_prior_mean".format(latent)])
            lengthscale_prior_std = float(config["kernel_params"]["kLengthscaleLatent{:d}_prior_std".format(latent)])
            def lengthscaleLogPrior(x, mean=lengthscale_prior_mean, std=lengthscale_prior_std):
                normal = torch.distributions.normal.Normal(loc=mean, scale=std)
                answer = normal.log_prob(x)
                return answer
        else:
            raise NotImplementedError("Prior for type {:s} for lengthscale has not been implemented yet".format(lengthscale_prior_type))

        period_prior_type = config["kernel_params"]["kPeriodLatent{:d}_prior_type".format(latent)]
        if period_prior_type=="normal":
            period_prior_mean = float(config["kernel_params"]["kPeriodLatent{:d}_prior_mean".format(latent)])
            period_prior_std = float(config["kernel_params"]["kPeriodLatent{:d}_prior_std".format(latent)])
            def periodLogPrior(x, mean=period_prior_mean, std=period_prior_std):
                normal = torch.distributions.normal.Normal(loc=mean, scale=std)
                answer = normal.log_prob(x)
                # pdb.set_trace()
                return answer
        else:
            raise NotImplementedError("Prior for type {:s} for period has not been implemented yet".format(lengthscale_prior_type))

    else:
        raise NotImplementedError("Prior for kernel {:s} has not been implemented yet".format(kernel_type))

    answer = lambda kernelParams: kernelParamsLogPrior(kernelParams=kernelParams, kernelScales=kernelScales, lengthscaleLogPrior=lengthscaleLogPrior, periodLogPrior=periodLogPrior)
    return answer


def _getEmbeddingLogPriors(config):
    def embeddingParamsLogPrior(embeddingParams, C_element_log_prior, d_element_log_prior):
        C = embeddingParams[0]
        ClogProb = 0.0
        for row in C:
            for value in row:
                ClogProb = ClogProb + C_element_log_prior(x=value)

        d = embeddingParams[1]
        dLogProb = 0.0
        for value in d:
            dLogProb = dLogProb + d_element_log_prior(x=value)

        answer = ClogProb + dLogProb
        # pdb.set_trace()
        return answer

    C_prior_type = config["embedding_params"]["C_prior_type"]
    if C_prior_type == "uniform":
        C_prior_min = float(config["embedding_params"]["C_prior_min"])
        C_prior_max = float(config["embedding_params"]["C_prior_max"])
        def C_element_log_prior(x, min=C_prior_min, max=C_prior_max):
            uniform = torch.distributions.uniform.Uniform(min, max)
            answer = uniform.log_prob(x)
            return answer
    else:
        raise NotImplementedError("Prior for type {:s} for embedding C matrix has not been implemented yet".format(C_prior_type))

    d_prior_type = config["embedding_params"]["d_prior_type"]
    if d_prior_type == "normal":
        d_prior_mean = float(config["embedding_params"]["d_prior_mean"])
        d_prior_std = float(config["embedding_params"]["d_prior_std"])
        def d_element_log_prior(x, mean=d_prior_mean, std=d_prior_std):
            normal = torch.distributions.normal.Normal(mean, std)
            answer = normal.log_prob(x)
            return answer
    else:
        raise NotImplementedError("Prior for type {:s} for embedding d vector has not been implemented yet".format(C_prior_type))

    answer = lambda embeddingParams: embeddingParamsLogPrior(embeddingParams=embeddingParams, C_element_log_prior=C_element_log_prior, d_element_log_prior=d_element_log_prior)
    return answer

def getVariationalMean0(nLatents, nTrials, config, keyNamePattern="qMu0Latent{:d}Trial{:d}_filename"):
    qMu0 = [[] for r in range(nLatents)]
    for k in range(nLatents):
        qMu0Filename = config["variational_params"][keyNamePattern.format(k, 0)]
        qMu0k0 = torch.from_numpy(pd.read_csv(qMu0Filename, header=None).to_numpy()).flatten()
        nIndPointsK = len(qMu0k0)
        qMu0[k] = torch.empty((nTrials, nIndPointsK, 1), dtype=torch.double)
        qMu0[k][0,:,0] = qMu0k0
        for r in range(1, nTrials):
            qMu0Filename = keyNamePattern.format(k, r)
            qMu0kr = pd.read_csv(qMu0Filename, header=None)
            qMu0[k][r,:,0] = qMu0kr
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
            qSigma0kr = pd.read_csv(qSigma0Filename, header=None)
            qSigma0[k][r,:,:] = qSigma0kr
    return qSigma0

def getScaledKernels(nLatents, config, forceUnitScale):
    kernels = [[] for r in range(nLatents)]
    kernelsParamsScales = [[] for r in range(nLatents)]
    for k in range(nLatents):
        kernelType = config["kernel_params"]["kTypeLatent{:d}".format(k)]
        if kernelType=="periodic":
            if not forceUnitScale:
                scale = float(config["kernel_params"]["kScaleValueLatent{:d}".format(k)])
            else:
                scale = 1.0
            lengthscaleScaledValue = float(config["kernel_params"]["kLengthscaleScaledValueLatent{:d}".format(k)])
            lengthscaleScale = float(config["kernel_params"]["kLengthscaleScaleLatent{:d}".format(k)])
            periodScaledValue = float(config["kernel_params"]["kPeriodScaledValueLatent{:d}".format(k)])
            periodScale = float(config["kernel_params"]["kPeriodScaleLatent{:d}".format(k)])
            kernel = stats.kernels.PeriodicKernel(scale=scale, lengthscaleScale=lengthscaleScale, periodScale=periodScale)
            kernel.setParams(params=torch.Tensor([lengthscaleScaledValue*lengthscaleScale, periodScaledValue*periodScale]).double())
            kernelsParamsScales[k] = torch.tensor([lengthscaleScale, periodScale])
        else:
            raise ValueError("Invalid kernel type {:s} for latent {:d}".format(kernelType, k))
        kernels[k] = kernel
    answer = {"kernels": kernels, "kernelsParamsScales": kernelsParamsScales}
    return answer

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
            kernel = stats.kernels.PeriodicKernel(scale=scale)
            kernel.setParams(params=torch.Tensor([lengthscale, period]).double())
        elif kernelType=="exponentialQuadratic":
            if not forceUnitScale:
                scale = float(config["kernel_params"]["kScaleValueLatent{:d}".format(k)])
            else:
                scale = 1.0
            lengthscale = float(config["kernel_params"]["kLengthscaleScaledValueLatent{:d}".format(k)])
            kernel = stats.kernels.ExponentialQuadraticKernel(scale=scale)
            kernel.setParams(params=torch.Tensor([lengthscale]).double())
        else:
            raise ValueError("Invalid kernel type {:s} for latent {:d}".format(kernelType, k))
        kernels[k] = kernel
    return kernels

def getQMu0(nLatents, nTrials, config):
    qMu0 = [[] for r in range(nLatents)]
    for k in range(nLatents):
        qMu0k0 = torch.tensor([float(str) for str in config["variational_params"]["qMu0Latent{:d}Trial0".format(k)][1:-1].split(", ")], dtype=torch.double)
        nIndPointsK = len(qMu0k0)
        qMu0[k] = torch.empty((nTrials, nIndPointsK, 1), dtype=torch.double)
        qMu0[k][0,:,0] = qMu0k0
        for r in range(1, nTrials):
            qMu0kr = torch.tensor([float(str) for str in config["variational_params"]["qMu0Latent{:d}Trial{:d}".format(k,r)][1:-1].split(", ")])
            qMu0[k][r,:,0] = qMu0kr
    return qMu0

def getIndPointsMeans(nTrials, nLatents, config):
    indPointsMeans = [[] for r in range(nTrials)]
    for r in range(nTrials):
        indPointsMeans[r] = [[] for k in range(nLatents)]
        for k in range(nLatents):
            indPointsMeans[r][k] = torch.tensor([float(str) for str in config["indPoints_params"]["indPointsMeanLatent{:d}Trial{:d}".format(k,r)][1:-1].split(", ")], dtype=torch.double).unsqueeze(dim=1)
    return indPointsMeans

def getIndPointsLocs0(nLatents, nTrials, config):
    Z0 = [[] for k in range(nLatents)]
    for k in range(nLatents):
        Z0_k_r0 = torch.tensor([float(str) for str in config["indPoints_params"]["indPointsLocsLatent{:d}Trial{:d}".format(k,0)][1:-1].split(", ")], dtype=torch.double)
        nIndPointsForLatent = len(Z0_k_r0)
        Z0[k] = torch.empty((nTrials, nIndPointsForLatent, 1), dtype=torch.double)
        Z0[k][0,:,0] = Z0_k_r0
        for r in range(1, nTrials):
            Z0[k][r,:,0] = torch.tensor([float(str) for str in config["indPoints_params"]["indPointsLocsLatent{:d}Trial{:d}".format(k,r)][1:-1].split(", ")], dtype=torch.double)
    return Z0

def getLatentsMeansFuncs(nLatents, nTrials, config):
    def getLatentMeanFunc(ampl, tau, freq, phase):
        mean = lambda t: ampl*torch.exp(-t/tau)*torch.sin(2*math.pi*freq*t + phase)
        return mean

    meansFuncs = [[] for r in range(nLatents)]
    for k in range(nLatents):
        ampl = float(config["latentMean_params"]["amplLatent{:d}".format(k)])
        tau = float(config["latentMean_params"]["tauLatent{:d}".format(k)])
        freq = float(config["latentMean_params"]["freqLatent{:d}".format(k)])
        phase = float(config["latentMean_params"]["phaseLatent{:d}".format(k)])
        meanFunc = getLatentMeanFunc(ampl=ampl, tau=tau, freq=freq, phase=phase)
        meansFuncs[k] = meanFunc
    return meansFuncs

def getLinearEmbeddingParams(CFilename, dFilename):
    df = pd.read_csv(CFilename, header=None)
    C = torch.from_numpy(df.values)
    df = pd.read_csv(dFilename, header=None)
    d = torch.from_numpy(df.values)
    # pdb.set_trace()
    return C, d

