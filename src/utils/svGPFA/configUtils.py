
import pdb
import math
import pandas as pd
import torch
import stats.kernels

def getKernels(nLatents, config):
    kernels = [[] for r in range(nLatents)]
    for k in range(nLatents):
        kernelType = config["kernel_params"]["kTypeLatent{:d}".format(k)]
        if kernelType=="periodic":
            # scale = float(config["kernel_params"]["kScaleLatent{:d}".format(k)])
            lengthScale = float(config["kernel_params"]["kLengthscaleLatent{:d}".format(k)])
            period = float(config["kernel_params"]["kPeriodLatent{:d}".format(k)])
            kernel = stats.kernels.PeriodicKernel(scale=1.0)
            kernel.setParams(params=torch.Tensor([lengthScale, period]))
        elif kernelType=="exponentialQuadratic":
            # scale = float(config["kernel_params"]["kScaleLatent{:d}".format(k)])
            lengthScale = float(config["kernel_params"]["kLengthscaleLatent{:d}".format(k)])
            kernel = stats.kernels.ExponentialQuadraticKernel(scale=1.0)
            kernel.setParams(params=torch.Tensor([lengthScale]))
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

def getIndPointsMeans(nIndPointsPerLatent, nTrials, config):
    nLatents = len(nIndPointsPerLatent)
    indPointsMeans = [[] for r in range(nTrials)]
    for r in range(nTrials):
        indPointsMeans[r] = [[] for k in range(nLatents)]
        for k in range(nLatents):
            indPointsMeans[r][k] = torch.tensor([float(str) for str in config["indPoints_params"]["indPointsMeanLatent{:d}Trial{:d}".format(k,r)][1:-1].split(", ")])
            if len(indPointsMeans[r][k])!=nIndPointsPerLatent[k]:
                   raise RuntimeError("Incorrect indPointsMeanLatent{:d}Trial{:d}".format(k,r))
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

def getLinearEmbeddingParams(nNeurons, nLatents, CFilename, dFilename):
    df = pd.read_csv(CFilename, header=None)
    C = torch.from_numpy(df.values)
    df = pd.read_csv(dFilename, header=None)
    d = torch.from_numpy(df.values)
    # pdb.set_trace()
    return C, d

