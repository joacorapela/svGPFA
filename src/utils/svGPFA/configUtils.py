
import pdb
import math
import pandas as pd
import torch
import stats.kernels

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
            kernel.setParams(params=torch.tensor([lengthscaleScaledValue*lengthscaleScale, periodScaledValue*periodScale], dtype=torch.double))
            kernelsParamsScales[k] = torch.tensor([lengthscaleScale, periodScale], dtype=torch.double)
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
            kernel.setParams(params=torch.tensor([lengthscale, period], dtype=torch.double))
        elif kernelType=="exponentialQuadratic":
            if not forceUnitScale:
                scale = float(config["kernel_params"]["kScaleValueLatent{:d}".format(k)])
            else:
                scale = 1.0
            lengthscale = float(config["kernel_params"]["kLengthscaleScaledValueLatent{:d}".format(k)])
            kernel = stats.kernels.ExponentialQuadraticKernel(scale=scale)
            kernel.setParams(params=torch.tensor([lengthscale], dtype=torch.double))
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

