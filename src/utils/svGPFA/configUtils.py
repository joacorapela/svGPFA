
def getKernels(nLatents, nTrials, config):
    kernels = [[] for r in range(nTrials)]
    for r in range(nTrials):
        kernels[r] = [[] for r in range(nLatents)]
        for k in range(nLatents):
            kernelType = config["kernel_params"]["kTypeLatent{:d}Trial{:d}".format(k, r)]
            if kernelType=="periodic":
                scale = float(config["kernel_params"]["kScaleLatent{:d}Trial{:d}".format(k, r)])
                lengthScale = float(config["kernel_params"]["kLengthscaleLatent{:d}Trial{:d}".format(k, r)])
                period = float(config["kernel_params"]["kPeriodLatent{:d}Trial{:d}".format(k, r)])
                kernel = stats.kernels.PeriodicKernel()
                kernel.setParams(params=torch.Tensor([scale, lengthScale, period]))
            elif kernelType=="exponentialQuadratic":
                scale = float(config["kernel_params"]["kScaleLatent{:d}Trial{:d}".format(k, r)])
                lengthScale = float(config["kernel_params"]["kLengthscaleLatent{:d}Trial{:d}".format(k, r)])
                kernel = stats.kernels.ExponentialQuadraticKernel()
                kernel.setParams(params=torch.Tensor([scale, lengthScale]))
            else:
                raise ValueError("Invalid kernel type {:s} for latent {:d} and trial {:d}".format(kernelType, k, r))
            kernels[r][k] = kernel
    return kernels

def getLatentsMeansFuncs(nLatents, nTrials, config):
    def getLatentMeanFunc(ampl, tau, freq, phase):
        mean = lambda t: ampl*torch.exp(-t/tau)*torch.cos(2*math.pi*freq*t + phase)
        return mean

    meansFuncs = [[] for r in range(nTrials)]
    for r in range(nTrials):
        meansFuncs[r] = [[] for r in range(nLatents)]
        for k in range(nLatents):
            ampl = float(config["latentMean_params"]["amplLatent{:d}Trial{:d}".format(k, r)])
            tau = float(config["latentMean_params"]["tauLatent{:d}Trial{:d}".format(k, r)])
            freq = float(config["latentMean_params"]["freqLatent{:d}Trial{:d}".format(k, r)])
            phase = float(config["latentMean_params"]["phaseLatent{:d}Trial{:d}".format(k, r)])
            meanFunc = getLatentMeanFunc(ampl=ampl, tau=tau, freq=freq, phase=phase)
            meansFuncs[r][k] = meanFunc
    return meansFuncs

def getLinearEmbeddingParams(nNeurons, nLatents, config):
    C = torch.DoubleTensor([float(str) for str in config["embedding_params"]["C"][1:-1].split(",")])
    C = torch.reshape(C, (nNeurons, nLatents))
    d = torch.DoubleTensor([float(str) for str in config["embedding_params"]["d"][1:-1].split(",")])
    d = torch.reshape(d, (nNeurons, 1))
    return C, d

