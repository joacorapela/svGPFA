
import pdb
import sys
import os
import math
import random
import torch
import matplotlib.pyplot as plt
import pickle
import configparser
sys.path.append("..")
import stats.svGPFA.simulations
import stats.kernels
import stats.gaussianProcesses.eval

def getLatents(nLatents, nTrials,
               k0Scale, k0LengthScale, k0Period,
               k1Scale, k1LengthScale, k1Period,
               k2Scale, k2LengthScale):
    latents = [[] for r in range(nTrials)]
    def meanLatentK0(t):
        answer = 0.8*torch.exp(-2*(torch.sin(math.pi*abs(t)*2.5)**2)/5)*torch.sin(2*math.pi*2.5*t)
        return answer
    def meanLatentK1(t):
        answer = 0.5*torch.cos(2*math.pi*2.5*t)
        return answer
    def meanLatentK2(t):
        answer = 0.7*torch.exp(-0.5*(t-5)**2/8)*torch.cos(2*math.pi*.12*t)+0.8*torch.exp(-0.5*(t-10)**2/12)*torch.sin(2*math.pi*t*0.1+1.5)
        return answer
    meanLatentsAll = [meanLatentK0, meanLatentK1, meanLatentK2]

    kernel0 = stats.kernels.PeriodicKernel()
    kernel0.setParams(params=[k0Scale, k0LengthScale, k0Period])
    kernel1 = stats.kernels.PeriodicKernel()
    kernel1.setParams(params=[k1Scale, k1LengthScale, k1Period])
    kernel2 = stats.kernels.ExponentialQuadraticKernel()
    kernel2.setParams(params=[k2Scale, k2LengthScale])
    kernelsAll = [kernel0, kernel1, kernel2]

    latentsForTrial = [[] for r in range(nLatents)]
    for k in range(nLatents):
        latentsForTrial[k] = stats.gaussianProcesses.eval.GaussianProcess(mean=meanLatentsAll[k], kernel=kernelsAll[k])
        # latentsForTrial[k] = meanLatentsAll[k]
    for r in range(nTrials):
        latents[r] = latentsForTrial
    return latents

def getLatentsSamples(latents, trialsLengths, dt):
    latentsSamples = [[] for r in range(len(latents))]
    for r in range(len(latents)):
        latentsSamples[r] = [[] for r in range(len(latents[r]))]
        for k in range(len(latents[r])):
            latentsSamples[r][k] = getLatentSamples(latent=latents[r][k],
                                                    trialLength=trialsLengths[r],
                                                    dt=dt)
    return latentsSamples

def getLatentSamples(latent, trialLength, dt):
    t = torch.arange(0, trialLength, dt)
    mean = latent.mean(t)
    std = latent.std(t)
    answer = {"t": t, "mean": mean, "std": std}
    return answer

def plotLatents(latents, latentsEpsilon, trialsLengths, dt, figFilename, alpha=0.5, 
                marker="x", xlabel="Time (sec)", ylabel="Amplitude"):
    nTrials = len(latents)
    nLatents = len(latents[0])
    f, axs = plt.subplots(nTrials,nLatents, sharex=True, sharey=True)
    for r in range(nTrials):
        t = torch.arange(0, trialsLengths[r], dt)
        for k in range(nLatents):
            mean = latents[r][k].mean(t)
            std = latents[r][k].std(t)
            axs[r,k].plot(t, latents[r][k](t, epsilon=latentsEpsilon), marker=marker)
            axs[r,k].fill_between(t, mean-1.96*std, mean+1.96*std, alpha=alpha)
            axs[r,k].set_xlabel(xlabel)
            axs[r,k].set_ylabel(ylabel)
            axs[r,k].set_title("r={}, k={}".format(r, k))
    plt.savefig(figFilename)

def plotSpikeTimes(spikeTimes, figFilename, trialToPlot, xlabel="Time (sec)",
                   ylabel="Neuron"):
    plt.eventplot(positions=spikeTimes[trialToPlot])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(figFilename)

def removeFirsSpike(spikeTimes):
    nTrials = len(spikeTimes)
    nNeurons = len(spikeTimes[0])
    sSpikeTimes = [[] for n in range(nTrials)]
    for r in range(nTrials):
        sSpikeTimes[r] = [[] for r in range(nNeurons)]
        for n in range(nNeurons):
            sSpikeTimes[r][n] = spikeTimes[r][n][1:]
    return(sSpikeTimes)

def main(argv):
    nNeurons = 50
    nTrials = 5
    nLatents = 3
    trialsLengths = [20]*nTrials
    dtSimulate = 1e-1
    dtLatentsFig = 1e-1
    spikeTrialToPlot = 0
    k0Scale, k0LengthScale, k0Period = 1, 1.5, 1/2.5
    k1Scale, k1LengthScale, k1Period = 1, 1.2, 1/2.5
    k2Scale, k2LengthScale = 1, 1
    latentsEpsilon = 1e-3

    randomPrefixUsed = True
    while randomPrefixUsed:
        randomPrefix = "{:08d}".format(random.randint(0, 10**8))
        metaDataFilename = \
            "results/{:s}_simulation_metaData.ini".format(randomPrefix)
        if not os.path.exists(metaDataFilename):
           randomPrefixUsed = False

    config = configparser.ConfigParser()
    config["latents_params"] = {"nLatents": nLatents,
                                "latentsEpsilon": latentsEpsilon}
    config["spikes_params"] = {"nTrials": nTrials,
                               "nNeurons": nNeurons}
    config["kernels_params"] = {"k0Scale": k0Scale,
                                "k0LengthScale": k0LengthScale,
                                "k0Period": k0Period,
                                "k1Scale": k1Scale,
                                "k1LengthScale": k1LengthScale,
                                "k1Period": k1Period,
                                "k2Scale": k2Scale,
                                "k2LengthScale": k2LengthScale}
    config["simulation_params"] = {"dt": dtSimulate,
                                   "trialLengths": trialsLengths}

    with open(metaDataFilename, "w") as f:
        config.write(f)

    latentsFilename = \
        "results/{:s}_simulation_latents.pickle".format(randomPrefix)
    spikeTimesFilename = \
        "results/{:s}_simulation_spikeTimes.pickle".format(randomPrefix)
    latentsFigFilename = \
        "figures/{:s}_simulation_latents.png".format(randomPrefix)
    spikeTimesFigFilename = \
        "figures/{:s}_simulation_spikeTimes.png".format(randomPrefix)

    latents = getLatents(nLatents=nLatents, nTrials=nTrials,
                         k0Scale=k0Scale, k0LengthScale=k0LengthScale, 
                         k0Period=k0Period,
                         k1Scale=k1Scale, k1LengthScale=k1LengthScale, 
                         k1Period=k1Period,
                         k2Scale=k2Scale, k2LengthScale=k2LengthScale)
    C = .4*torch.randn(size=(nNeurons, nLatents))*torch.tensor([1, 1.2, 1.3])
    d = -.1*torch.ones(nNeurons)
    simulator = stats.svGPFA.simulations.GPFASimulator()
    spikeTimes = simulator.simulate(nNeurons=nNeurons,
                                    trialsLengths=trialsLengths,
                                    latents=latents, C=C, d=d,
                                    linkFunction=torch.exp, dt=dtSimulate, 
                                    latentsEpsilon=latentsEpsilon)

    latentsSamples = getLatentsSamples(latents=latents, 
                                       trialsLengths=trialsLengths, 
                                       dt=dtSimulate)

    with open(latentsFilename, "wb") as f: pickle.dump(latentsSamples, f)
    with open(spikeTimesFilename, "wb") as f: pickle.dump(spikeTimes, f)

    plotLatents(latents=latents, latentsEpsilon=latentsEpsilon, trialsLengths=trialsLengths, dt=dtLatentsFig, figFilename=latentsFigFilename)
    plt.figure()
    plotSpikeTimes(spikeTimes=spikeTimes, trialToPlot=spikeTrialToPlot, figFilename=spikeTimesFigFilename)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
