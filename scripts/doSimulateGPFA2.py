
import pdb
import sys
import os
import math
import random
import torch
import matplotlib.pyplot as plt
import pickle
import configparser
sys.path.append("../src")
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

def plotLatents(latents, latentsEpsilon, trialsLengths, dt, figFilename, alpha=0.5, marker="x", xlabel="Time (sec)", ylabel="Amplitude"):
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
    return sSpikeTimes

def buildKernels(nLatents, nTrials, config):
    kernels = []
    for k in range(nLatents):
        kernelsForLatent = []
        for r in range(nTrials):
            kernelType = simConfig["kernel_params"]["kTypeLatent{:d}Trial{:d}".format(k, r)]
            if kernelType=="periodic":
                scale = float(kernelType = simConfig["kernel_params"]["kScaleLatent{:d}Trial{:d}".format(k, r)])
                lengthscale = float(kernelType = simConfig["kernel_params"]["kLengthscaleLatent{:d}Trial{:d}".format(k, r)])
                period = float(kernelType = simConfig["kernel_params"]["kPeriodLatent{:d}Trial{:d}".format(k, r)])
                kernel = stats.kernels.PeriodicKernel()
                kernel.setParams(params=[scale, lengthScale, period])
            elif kernelType=="exponentialQuadratic":
                scale = float(kernelType = simConfig["kernel_params"]["kScaleLatent{:d}Trial{:d}".format(k, r)])
                lengthscale = float(kernelType = simConfig["kernel_params"]["kLengthscaleLatent{:d}Trial{:d}".format(k, r)])
                kernel = stats.kernels.ExponentialQuadraticKernel()
                kernel.setParams(params=[scale, lengthScale])
            else:
                raise ValueError("Invalid kernel type {:s} for latent {:d} and trial {:d}".format(kernelType, k, r))
            kernelsForLatent.append(kernel)
        kernels.append(kernelsForLatent)
    return kernels

def buildLatents(times, indPointsLocs, kernels, svPosteriorOnIndPointsParams,
                 kzzEpsilon=1e-5, dtype=):
    # times is a list of size nTrials
    # times[r] contains the times for trial r
    nLatents = len(kernels)
    nTrials = len(kernels[0])
    kzzKMS = IndPointsLocsKMS(epsilon=kzzEpsilon)
    kztKMS = IndPointsLocsAndAllTimesKMS()
    svPosteriorOnIndPoints = SVPosteriorOnIndPoints()
    svPosteriorOnIndPoints.setInitialParams(initialParams=svPosteriorOnIndPointsParams)
    svPosteriorOnLatents = svPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=svPosteriorOnIndPoints,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndTimesKMS)
    svPosteriorOnLatents.setKernels(kernels=kernels)
    svPosteriorOnLatents.setIndPointsLocs(indPointsLocs=indPointsLocs)
    svPosteriorOnLatents.setTimes(times=times)
    svPosteriorOnLatents.buildKernelMatrices()
    qKMu, qKVar = svPosteriorOnLatents.computeMeansAndVars()
    latents = [[] for r in range(nTrials)]
    for r in range(nTrials):
        latentsForTrial = torch.new_empty((len(times[r]), nLatents),
                                          dtype=indPointsLocs[0].dtype, 
                                          device=indPointsLocs[0].device)
        for k in range(nLatents):
            mn = scipy.stats.multivariate_normal(mean=qKMu[k][r,:,0], cov=qKVar)
            latentesForTrial[:,k] = 
            latentsForTrial[k] = stats.gaussianProcesses.eval.GaussianProcess(mean=meanLatentsAll[k], kernel=kernelsAll[k])

def getSVOnIndPointsParams(nLatents, nTrials, config):
    qParams = []
    for k in range(nLatents):
        qParamsForLatent = []
        for r in range(nTrials):
            qMu = torch.FloatTensor([float(str) for str in simConfig["svPosteriorIndPoints_params"]["qMuLatent{:d}Trial{:d}".format(k, r)][1:-1].split(",")])
            qSVec = torch.FloatTensor([float(str) for str in simConfig["svPosteriorIndPoints_params"]["qSVecLatent{:d}Trial{:d}".format(k, r)][1:-1].split(",")])
            qSDiag = torch.FloatTensor([float(str) for str in simConfig["svPosteriorIndPoints_params"]["qSDiagLatent{:d}Trial{:d}".format(k, r)][1:-1].split(",")])
            qParams = {"qMu": qMu, "qSVec": qSVec, "qSDiag": qSDiag}
            qParamsForLatent.append(qParams)
        qParams.append(qParamsForLatent)
    return qParams

def main(argv):
    simPrefix = "00000002_simulation"
    paramsFilename = "data/{:s}_metaData.ini".format(simPrefix)
    simConfig = configparser.ConfigParser()
    simConfig.read(simConfigFilename)
    nLatents = int(simConfig["control_variables"]["nLatents"])
    nNeurons = int(simConfig["control_variables"]["nNeurons"])
    trialLengths = torch.IntTensor([int(str) for str in simConfig["control_variables"]["trialLengths"][1:-1].split(",")])
    nTrials = len(nTrialLengths)
    dtSimulate = int(simConfig["control_variables"]["dt"])
    dtLatentsFig = 1e-1
    spikeTrialToPlot = 0
    latentsEpsilon = 1e-3

    indPointLocs = getInducingPointLocs(nLatents=nLatents, nTrials=nTrials, config=simConfig)
    kernels = buildKernels(nLatents=nLatents, nTrials=nTrials, config=simConfig)
    qParams = getSVOnIndPointsParams(nLatents=nLatents, nTrials=nTrials, config=simConfig)
    latents = buildLatents(indPointLocs=indPointLocs, kernels=kernels, svOnIndPointsParams=qParams)

    kernels = []
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
