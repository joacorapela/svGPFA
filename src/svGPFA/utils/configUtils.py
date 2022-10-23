
import math
import numpy as np
import torch
import svGPFA.stats.kernels


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


def getScaledKernels(nLatents, config, forceUnitScale):
    kernels = [[] for r in range(nLatents)]
    kernelsParamsScales = [[] for r in range(nLatents)]
    for k in range(nLatents):
        kernelType = config["kernel_params"]["kTypeLatent{:d}".format(k)]
        if kernelType == "periodic":
            if not forceUnitScale:
                scale = float(config["kernel_params"]["kScaleValueLatent{:d}".format(k)])
            else:
                scale = 1.0
            lengthscaleScaledValue = float(config["kernel_params"]["kLengthscaleScaledValueLatent{:d}".format(k)])
            lengthscaleScale = float(config["kernel_params"]["kLengthscaleScaleLatent{:d}".format(k)])
            periodScaledValue = float(config["kernel_params"]["kPeriodScaledValueLatent{:d}".format(k)])
            periodScale = float(config["kernel_params"]["kPeriodScaleLatent{:d}".format(k)])
            kernel = svGPFA.stats.kernels.PeriodicKernel(scale=scale, lengthscaleScale=lengthscaleScale, periodScale=periodScale)
            kernel.setParams(params=torch.Tensor([lengthscaleScaledValue*lengthscaleScale, periodScaledValue*periodScale]).double())
            kernelsParamsScales[k] = torch.tensor([lengthscaleScale, periodScale])
        else:
            raise ValueError("Invalid kernel type {:s} for latent {:d}".format(kernelType, k))
        kernels[k] = kernel
    answer = {"kernels": kernels, "kernelsParamsScales": kernelsParamsScales}
    return answer


def getVariationalMean0FromList(nLatents, nTrials, config):
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
            indPointsMeansFN = config["indPoints_params"]["indPointsMeanFNLatent{:d}Trial{:d}".format(k,r)]
            indPointsMeans[r][k] = torch.reshape(torch.from_numpy(np.loadtxt(indPointsMeansFN)), (-1,1))
    return indPointsMeans


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


