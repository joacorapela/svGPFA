
import pdb
import sys
import os
import random
import torch
import plotly
import matplotlib.pyplot as plt
import pickle
import configparser
sys.path.append("../src")
import simulations.svGPFA.simulations
import stats.gaussianProcesses.eval
from utils.svGPFA.configUtils import getKernels, getLatentsMeansFuncs, getLinearEmbeddingParams
from utils.svGPFA.miscUtils import getLatentsSamples, getTrialsTimes
import plot.svGPFA.plotUtils
from stats.pointProcess.tests import KSTestTimeRescalingUnbinned

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

def main(argv):
    if len(argv)!=2:
        print("Usage {:s} <simulation config number>".format(argv[0]))
        return
    trialKSTestTimeRescaling = 0
    neuronKSTestTimeRescaling = 0

    # load data and initial values
    simConfigNumber = int(argv[1])
    simConfigFilename = "data/{:08d}_simulation_metaData.ini".format(simConfigNumber)
    simConfig = configparser.ConfigParser()
    simConfig.read(simConfigFilename)
    nLatents = int(simConfig["control_variables"]["nLatents"])
    nNeurons = int(simConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    T = torch.tensor(trialsLengths).max()
    dtSimulate = float(simConfig["control_variables"]["dt"])
    dtLatentsFig = 1e-1
    gpRegularization = 1e-3

    randomPrefixUsed = True
    while randomPrefixUsed:
        randomPrefix = "{:08d}".format(random.randint(0, 10**8))
        metaDataFilename = \
            "results/{:s}_simulation_metaData.ini".format(randomPrefix)
        if not os.path.exists(metaDataFilename):
           randomPrefixUsed = False
    simResFilename = "results/{:s}_simRes.pickle".format(randomPrefix)
    latentsFigFilename = \
        "figures/{:s}_simulation_latents.png".format(randomPrefix)
    spikesTimesFigFilename = \
        "figures/{:s}_simulation_spikesTimes.png".format(randomPrefix)
    ksTestTimeRescalingFigFilename = \
        "figures/{:s}_simulation_ksTestTimeRescaling.png".format(randomPrefix)

    with torch.no_grad():
        kernels = getKernels(nLatents=nLatents, nTrials=nTrials, config=simConfig)
        latentsMeansFuncs = getLatentsMeansFuncs(nLatents=nLatents, nTrials=nTrials, config=simConfig)
        trialsTimes = getTrialsTimes(trialsLengths=trialsLengths, dt=dtSimulate)
        print("Computing latents samples")
        C, d = getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, config=simConfig)
        latentsSamples = getLatentsSamples(meansFuncs=latentsMeansFuncs,
                                           kernels=kernels,
                                           trialsTimes=trialsTimes,
                                           gpRegularization=gpRegularization,
                                           dtype=C.dtype)

        simulator = simulations.svGPFA.simulations.GPFASimulator()
        res = simulator.simulate(trialsTimes=trialsTimes, latentsSamples=latentsSamples, C=C, d=d, linkFunction=torch.exp)
        spikesTimes = res["spikesTimes"]
        intensityTimes = res["intensityTimes"]
        intensityValues = res["intensityValues"]
        latentsMeans, latentsSTDs = getLatentsMeansAndSTDs(meansFuncs=latentsMeansFuncs, kernels=kernels, trialsTimes=trialsTimes)

    simRes = {"times": trialsTimes, "latents": latentsSamples,
              "latentsMeans": latentsMeans, "latentsSTDs": latentsSTDs,
              "spikes": spikesTimes}
    with open(simResFilename, "wb") as f: pickle.dump(simRes, f)

    simResConfig = configparser.ConfigParser()
    simResConfig["simulation_params"] = {"simInitConfigFilename": simConfigFilename}
    simResConfig["simulation_results"] = {"simResFilename": simResFilename}
    with open(metaDataFilename, "w") as f:
        simResConfig.write(f)

    pLatents = plot.svGPFA.plotUtils.getSimulatedLatentsPlot(trialsTimes=trialsTimes, latentsSamples=latentsSamples, latentsMeans=latentsMeans, latentsSTDs=latentsSTDs, figFilename=latentsFigFilename)
    # pLatents = ggplotly(pLatents)
    # pLatents.show()

    pSpikes = plot.svGPFA.plotUtils.getSimulatedSpikeTimesPlot(spikesTimes=spikesTimes, figFilename=spikesTimesFigFilename)
    # pSpikes = ggplotly(pSpikes)
    # pSpikes.show()

    spikesTimesKS = spikesTimes[trialKSTestTimeRescaling][neuronKSTestTimeRescaling]
    cifKS = intensityValues[trialKSTestTimeRescaling][neuronKSTestTimeRescaling]
    sUTRISIs, uCDF, cb = KSTestTimeRescalingUnbinned(spikesTimes=spikesTimesKS,
                                                     cif=cifKS, t0=0, tf=T,
                                                     dt=dtSimulate)
    title = "Trial {:d}, Neuron {:d}".format(trialKSTestTimeRescaling, neuronKSTestTimeRescaling)
    plot.svGPFA.plotUtils.plotResKSTestTimeRescaling(sUTRISIs=sUTRISIs,
                                                     uCDF=uCDF, cb=cb,
                                                     figFilename=ksTestTimeRescalingFigFilename,
                                                     title=title)
    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
