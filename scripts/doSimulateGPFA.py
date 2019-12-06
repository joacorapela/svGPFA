
import pdb
import sys
import math
import torch
import matplotlib.pyplot as plt
import pickle
sys.path.append("../src")
import simulations
import kernels
import stats.gaussianProcesses.core

def getLatents(k0Scale=.1, k0LengthScale=1.5, k0Period=1/2.5,
               k1Scale=.1, k1LengthScale=1.2, k1Period=1/2.5,
               k2Scale=.1, k2LengthScale=1):
    K = 3
    R = 5

    latents = [[] for r in range(R)]
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

    kernel0 = kernels.PeriodicKernel()
    kernel0.setParams(params=[k0Scale, k0LengthScale, k0Period])
    kernel1 = kernels.PeriodicKernel()
    kernel1.setParams(params=[k1Scale, k1LengthScale, k1Period])
    kernel2 = kernels.ExponentialQuadraticKernel()
    kernel2.setParams(params=[k2Scale, k2LengthScale])
    kernelsAll = [kernel0, kernel1, kernel2]

    latentsForTrial = [[] for r in range(K)]
    for k in range(K):
        latentsForTrial[k] = stats.gaussianProcesses.core.GaussianProcess(mean=meanLatentsAll[k], kernel=kernelsAll[k])
        # latentsForTrial[k] = meanLatentsAll[k]
    for r in range(R):
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

def plotLatents(latents, trialsLengths, dt, figFilename, alpha=0.5, marker="x",
               xlabel="Time (sec)", ylabel="Amplitude"):
    R = len(latents)
    K = len(latents[0])
    f, axs = plt.subplots(R,K, sharex=True, sharey=True)
    for r in range(R):
        t = torch.arange(0, trialsLengths[r], dt)
        for k in range(K):
            mean = latents[r][k].mean(t)
            std = latents[r][k].std(t)
            axs[r,k].plot(t, latents[r][k](t), marker=marker)
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

def main(argv):
    nNeurons = 50
    nTrial = 5
    trialsLengths = [20]*nTrial
    dtSimulate = 1e-2
    dtLatentsFig = 1e-2
    # dtSimulate = 1e-1
    # dtLatentsFig = 1e-1
    spikeTrialToPlot = 0
    k0Scale, k0LengthScale, k0Period = 0.1, 1.5, 1/2.5
    k1Scale, k1LengthScale, k1Period =.1, 1.2, 1/2.5,
    k2Scale, k2LengthScale = .1, 1
    latentsPickleFilename = "results/latents_k0Scale{:.2f}_k0LengthScale{:.2f}_k0Period{:.2f}_k1Scale{:.2f}_k1LengthScale{:.2f}_k1Period{:.2f}_k2Scale{:.2f}_k2LengthScale{:.2f}.pickle".format(k0Scale, k0LengthScale, k0Period, k1Scale, k1LengthScale, k1Period, k2Scale, k2LengthScale)
    spikeTimesPickleFilename = "results/spikeTimes_k0Scale{:.2f}_k0LengthScale{:.2f}_k0Period{:.2f}_k1Scale{:.2f}_k1LengthScale{:.2f}_k1Period{:.2f}_k2Scale{:.2f}_k2LengthScale{:.2f}.pickle".format(k0Scale, k0LengthScale, k0Period, k1Scale, k1LengthScale, k1Period, k2Scale, k2LengthScale)
    latentsFigFilename = "figures/latents_k0Scale{:.2f}_k0LengthScale{:.2f}_k0Period{:.2f}_k1Scale{:.2f}_k1LengthScale{:.2f}_k1Period{:.2f}_k2Scale{:.2f}_k2LengthScale{:.2f}.png".format(k0Scale, k0LengthScale, k0Period, k1Scale, k1LengthScale, k1Period, k2Scale, k2LengthScale)
    spikeTimesFigFilename = "figures/spikeTimes_k0Scale{:.2f}_k0LengthScale{:.2f}_k0Period{:.2f}_k1Scale{:.2f}_k1LengthScale{:.2f}_k1Period{:.2f}_k2Scale{:.2f}_k2LengthScale{:.2f}.png".format(k0Scale, k0LengthScale, k0Period, k1Scale, k1LengthScale, k1Period, k2Scale, k2LengthScale)

    latents = getLatents(k0Scale=.1, k0LengthScale=1.5, k0Period=1/2.5,
                         k1Scale=.1, k1LengthScale=1.2, k1Period=1/2.5,
                         k2Scale=.1, k2LengthScale=1)
    nLatents = len(latents[0])
    C = .4*torch.randn(size=(nNeurons, nLatents))*torch.tensor([1, 1.2, 1.3])
    d = -.1*torch.ones(nNeurons)
    simulator = simulations.GPFASimulator()
    spikeTimes = simulator.simulate(nNeurons=nNeurons,
                                    trialsLengths=trialsLengths,
                                    latents=latents, C=C, d=d,
                                    linkFunction=torch.exp, dt=dtSimulate)

    latentsSamples = getLatentsSamples(latents=latents, trialsLengths=trialsLengths, dt=dtSimulate)

    with open(latentsPickleFilename, "wb") as f: pickle.dump(latentsSamples, f)
    with open(spikeTimesPickleFilename, "wb") as f: pickle.dump(spikeTimes, f)

    plotLatents(latents=latents, trialsLengths=trialsLengths, dt=dtLatentsFig,
                figFilename=latentsFigFilename)
    plt.figure()
    plotSpikeTimes(spikeTimes=spikeTimes, trialToPlot=spikeTrialToPlot,
                   figFilename=spikeTimesFigFilename)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
