
import pdb
import sys
import os
import math
import random
import torch
import plotly
import matplotlib.pyplot as plt
import pickle
import configparser
sys.path.append("../src")
import stats.svGPFA.simulations
import stats.kernels
import stats.gaussianProcesses.eval

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

def getLatentsSamples(meansFuncs, kernels, trialsTimes, latentsEpsilon, dtype):
    nTrials = len(kernels)
    nLatents = len(kernels[0])
    latentsSamples = [[] for r in range(nTrials)]

    for r in range(nTrials):
        print("Procesing trial {:d}".format(r))
        latentsSamples[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
        for k in range(nLatents):
            print("Procesing latent {:d}".format(k))
            gp = stats.gaussianProcesses.eval.GaussianProcess(mean=meansFuncs[r][k], kernel=kernels[r][k])
            latentsSamples[r][k,:] = gp.eval(t=trialsTimes[r], epsilon=latentsEpsilon)
    return latentsSamples

def getLatentsMeansAndSTDs(meansFuncs, kernels, trialsTimes):
    nTrials = len(kernels)
    nLatents = len(kernels[0])
    latentsMeans = [[] for r in range(nTrials)]
    latentsSTDs = [[] for r in range(nTrials)]

    for r in range(nTrials):
        latentsMeans[r] = torch.empty((nLatents, len(trialsTimes[r])))
        latentsSTDs[r] = torch.empty((nLatents, len(trialsTimes[r])))
        for k in range(nLatents):
            gp = stats.gaussianProcesses.eval.GaussianProcess(mean=meansFuncs[r][k], kernel=kernels[r][k])
            latentsMeans[r][k,:] = gp.mean(t=trialsTimes[r])
            latentsSTDs[r][k,:] = gp.std(t=trialsTimes[r])
    return latentsMeans, latentsSTDs

def getLatentsTimes(trialsLengths, dt):
    nTrials = len(trialsLengths)
    latentsTimes = [[] for r in range(nTrials)]
    for r in range(nTrials):
        latentsTimes[r] = torch.linspace(0, trialsLengths[r], round(trialsLengths[i]/dt))
    return latentsTimes

def getLinearEmbeddingParams(nNeurons, nLatents, config):
    C = torch.DoubleTensor([float(str) for str in config["embedding_params"]["C"][1:-1].split(",")])
    C = torch.reshape(C, (nNeurons, nLatents))
    d = torch.DoubleTensor([float(str) for str in config["embedding_params"]["d"][1:-1].split(",")])
    d = torch.reshape(d, (nNeurons, 1))
    return C, d

def getTrialsTimes(trialsLengths, dt):
    nTrials = len(trialsLengths)
    trialsTimes = [[] for r in range(nTrials)]
    for r in range(nTrials):
        trialsTimes[r] = torch.linspace(0, trialsLengths[r], round(trialsLengths[r]/dt))
    return trialsTimes

def plotLatents(trialsTimes, latentsSamples, latentsMeans, latentsSTDs, figFilename, alpha=0.5, marker="x", xlabel="Time (sec)", ylabel="Amplitude"):
    nTrials = len(latentsSamples)
    nLatents = latentsSamples[0].shape[0]
    f, axs = plt.subplots(nTrials, nLatents, sharex=True, sharey=True)
    for r in range(nTrials):
        t = trialsTimes[r]
        for k in range(nLatents):
            latentSamples = latentsSamples[r][k,:]
            mean = latentsMeans[r][k,:]
            std = latentsSTDs[r][k,:]
            axs[r,k].plot(t, latentSamples, marker=marker)
            axs[r,k].fill_between(t, mean-1.96*std, mean+1.96*std, alpha=alpha)
            axs[r,k].set_xlabel(xlabel)
            axs[r,k].set_ylabel(ylabel)
            axs[r,k].set_title("r={}, k={}".format(r, k))
    plt.savefig(figFilename)
    return f

def plotSpikeTimes(spikesTimes, figFilename, trialToPlot, xlabel="Time (sec)",
                   ylabel="Neuron"):
    plt.eventplot(positions=spikesTimes[trialToPlot])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(figFilename)
    f = plt.gcf()
    return f

def main(argv):
    simPrefix = "00000003_simulation"
    simConfigFilename = "data/{:s}_metaData.ini".format(simPrefix)
    simConfig = configparser.ConfigParser()
    simConfig.read(simConfigFilename)
    nLatents = int(simConfig["control_variables"]["nLatents"])
    nNeurons = int(simConfig["control_variables"]["nNeurons"])
    trialsLengths = [int(str) for str in simConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    dtSimulate = float(simConfig["control_variables"]["dt"])
    dtLatentsFig = 1e-1
    spikeTrialToPlot = 0
    latentsEpsilon = 1e-3

    randomPrefixUsed = True
    while randomPrefixUsed:
        randomPrefix = "{:08d}".format(random.randint(0, 10**8))
        metaDataFilename = \
            "results/{:s}_metaData.ini".format(randomPrefix)
        if not os.path.exists(metaDataFilename):
           randomPrefixUsed = False
    simResFilename = "results/{:s}_simRes.pickle".format(randomPrefix)
    latentsFigFilename = \
        "figures/{:s}_simulation_latents.png".format(randomPrefix)
    spikeTimesFigFilename = \
        "figures/{:s}_simulation_spikeTimes.png".format(randomPrefix)

    with open(metaDataFilename, "w") as f:
        simConfig.write(f)

    with torch.no_grad():
        kernels = getKernels(nLatents=nLatents, nTrials=nTrials, config=simConfig)
        latentsMeansFuncs = getLatentsMeansFuncs(nLatents=nLatents, nTrials=nTrials, config=simConfig)
        trialsTimes = getTrialsTimes(trialsLengths=trialsLengths, dt=dtSimulate)
        print("Computing latents samples")
        C, d = getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, config=simConfig)
        latentsSamples = getLatentsSamples(meansFuncs=latentsMeansFuncs,
                                           kernels=kernels,
                                           trialsTimes=trialsTimes,
                                           latentsEpsilon=latentsEpsilon,
                                           dtype=C.dtype)

        simulator = stats.svGPFA.simulations.GPFASimulator()
        spikesTimes = simulator.simulate(trialsTimes=trialsTimes,
                                         latentsSamples=latentsSamples,
                                         C=C, d=d, linkFunction=torch.exp)
        latentsMeans, latentsSTDs = getLatentsMeansAndSTDs(meansFuncs=latentsMeansFuncs, kernels=kernels, trialsTimes=trialsTimes)

    simRes = {"times": trialsTimes, "latents": latentsSamples, 
              "latentsMeans": latentsMeans, "latentsSTDs": latentsSTDs, 
              "spikes": spikesTimes}
    with open(simResFilename, "wb") as f: pickle.dump(simRes, f)

    pLatents = plotLatents(trialsTimes=trialsTimes, latentsSamples=latentsSamples, latentsMeans=latentsMeans, latentsSTDs=latentsSTDs, figFilename=latentsFigFilename)
    # pLatents = ggplotly(pLatents)
    # pLatents.show()

    plt.figure()

    pSpkes = plotSpikeTimes(spikesTimes=spikesTimes, trialToPlot=spikeTrialToPlot, figFilename=spikeTimesFigFilename)
    # pSpikes = ggplotly(pSpikes)
    # pSpikes.show()

    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
