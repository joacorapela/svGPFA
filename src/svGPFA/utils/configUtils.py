
import math
import numpy as np
import svGPFA.stats.kernels


def getKernels(n_latents, config, force_unit_scale):
    kernels = [[] for r in range(n_latents)]
    for k in range(n_latents):
        kernelType = config["kernel_params"]["kTypeLatent{:d}".format(k)]
        if kernelType=="periodic":
            if not force_unit_scale:
                scale = float(config["kernel_params"]["kScaleValueLatent{:d}".format(k)])
            else:
                scale = 1.0
            lengthscale = float(config["kernel_params"]["kLengthscaleScaledValueLatent{:d}".format(k)])
            period = float(config["kernel_params"]["kPeriodScaledValueLatent{:d}".format(k)])
            kernel = svGPFA.stats.kernels.PeriodicKernel(scale=scale)
            kernel.setParams(params=torch.Tensor([lengthscale, period]).double())
        elif kernelType=="exponentialQuadratic":
            if not force_unit_scale:
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


def getScaledKernels(n_latents, config, force_unit_scale):
    kernels = [[] for r in range(n_latents)]
    kernelsParamsScales = [[] for r in range(n_latents)]
    for k in range(n_latents):
        kernelType = config["kernel_params"]["kTypeLatent{:d}".format(k)]
        if kernelType == "periodic":
            if not force_unit_scale:
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


def getVariationalMean0FromList(n_latents, n_trials, config):
    qMu0 = [[] for r in range(n_latents)]
    for k in range(n_latents):
        qMu0k0 = torch.tensor([float(str) for str in config["variational_params"]["qMu0Latent{:d}Trial0".format(k)][1:-1].split(", ")], dtype=torch.double)
        nIndPointsK = len(qMu0k0)
        qMu0[k] = torch.empty((n_trials, nIndPointsK, 1), dtype=torch.double)
        qMu0[k][0,:,0] = qMu0k0
        for r in range(1, n_trials):
            qMu0kr = torch.tensor([float(str) for str in config["variational_params"]["qMu0Latent{:d}Trial{:d}".format(k,r)][1:-1].split(", ")])
            qMu0[k][r,:,0] = qMu0kr
    return qMu0


def getIndPointsMeans(n_trials, n_latents, config):
    indPointsMeans = [[] for r in range(n_trials)]
    for r in range(n_trials):
        indPointsMeans[r] = [[] for k in range(n_latents)]
        for k in range(n_latents):
            indPointsMeansFN = config["indPoints_params"]["indPointsMeanFNLatent{:d}Trial{:d}".format(k,r)]
            indPointsMeans[r][k] = torch.reshape(torch.from_numpy(np.loadtxt(indPointsMeansFN)), (-1,1))
    return indPointsMeans


def getLatentsMeansFuncs(n_latents, n_trials, config):
    def getLatentMeanFunc(ampl, tau, freq, phase):
        mean = lambda t: ampl*torch.exp(-t/tau)*torch.sin(2*math.pi*freq*t + phase)
        return mean

    meansFuncs = [[] for r in range(n_latents)]
    for k in range(n_latents):
        ampl = float(config["latentMean_params"]["amplLatent{:d}".format(k)])
        tau = float(config["latentMean_params"]["tauLatent{:d}".format(k)])
        freq = float(config["latentMean_params"]["freqLatent{:d}".format(k)])
        phase = float(config["latentMean_params"]["phaseLatent{:d}".format(k)])
        meanFunc = getLatentMeanFunc(ampl=ampl, tau=tau, freq=freq, phase=phase)
        meansFuncs[k] = meanFunc
    return meansFuncs


