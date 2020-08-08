
import pdb
import sys
import os
import random
import torch
import pickle
import argparse
import configparser
import matplotlib.pyplot as plt
sys.path.append("../src")
import simulations.svGPFA.simulations
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import utils.svGPFA.initUtils
import stats.svGPFA.svPosteriorOnLatents
import stats.svGPFA.svPosteriorOnIndPoints
import stats.svGPFA.kernelsMatricesStore

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simInitConfigNumber", help="Simulation initialization configuration number", type=int)
    parser.add_argument("--latentTrialToPlot", help="Trial on which to plot the Latent", type=int, default=0)
    parser.add_argument("--cifTrialToPlot", help="Trial on which to plot the CIF", type=int, default=0)
    parser.add_argument("--cifNeuronToPlot", help="Neuron on which to plot the CIF", type=int, default=5)
    args = parser.parse_args()

    simInitConfigNumber = args.simInitConfigNumber
    latentTrialToPlot = args.latentTrialToPlot
    cifTrialToPlot = args.cifTrialToPlot
    cifNeuronToPlot = args.cifNeuronToPlot

    # load data and initial values
    simInitConfigFilename = "data/{:08d}_simulation_metaData.ini".format(simInitConfigNumber)
    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nLatents = int(simInitConfig["control_variables"]["nLatents"])
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    T = torch.tensor(trialsLengths).max().item()
    dtCIF = float(simInitConfig["control_variables"]["dtCIF"])
    latentsGPRegularizationEpsilon = float(simInitConfig["control_variables"]["latentsGPRegularizationEpsilon"])
    firstIndPointLoc = float(simInitConfig["control_variables"]["firstIndPointLoc"])

    randomPrefixUsed = True
    while randomPrefixUsed:
        simNumber = random.randint(0, 10**8)
        simResConfigFilename = \
            "results/{:08d}_simulation_metaData.ini".format(simNumber)
        if not os.path.exists(simResConfigFilename):
           randomPrefixUsed = False
    simResFilename = "results/{:08d}_simRes.pickle".format(simNumber)

    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig)
    kernelsParams0 = utils.svGPFA.initUtils.getKernelsParams0(kernels=kernels, noiseSTD=0.0)
#     qMu0 = utils.svGPFA.configUtils.getQMu0(nTrials=nTrials, nLatents=nLatents, config=simInitConfig)
#     qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
#     indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS()
#     indPointsLocsKMS.setEpsilon(epsilon=latentsGPRegularizationEpsilon)
#     indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
#     qKAllTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
#         svPosteriorOnIndPoints=qU,
#         indPointsLocsKMS=indPointsLocsKMS,
#         indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS
#     )
#     qKAllTimes.setKernels(kernels=kernels)
#     nIndPointsPerLatent = [qMu0[r].shape[1] for r in range(len(qMu0))]
#     Z0 = utils.svGPFA.initUtils.getIndPointLocs0(nIndPointsPerLatent=nIndPointsPerLatent, trialsLengths=trialsLengths, firstIndPointLoc=firstIndPointLoc)
#     qUParams0 = {"qMu0": qMu0, "qSVec0": qMu0, "qSDiag0": qMu0}
#     kmsParams0 = {"kernelsParams0": kernelsParams0,
#                   "inducingPointsLocs0": Z0}
#     qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
#                  "kernelsMatricesStore": kmsParams0}
#     qKAllTimes.setInitialParams(initialParams=qKParams0)
#     latentsMeansFuncs = utils.svGPFA.configUtils.getLatentsMeansFuncs(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
#     latentsMeansFuncs = utils.svGPFA.miscUtils.getSVGPFALatentsMeansFuncs(kernels=kernels, scales=svGPFALatentsScales)
    cifTrialsTimes = utils.svGPFA.miscUtils.getTrialsTimes(trialsLengths=trialsLengths, dt=dtCIF)
    cifTrialsTimesMatrix = torch.empty((nTrials, len(cifTrialsTimes[0]), 1))
    # patch to accomodate unreasonable need to have trial times of equal length
    # across trials
#     for r in range(nTrials):
#         cifTrialsTimesMatrix[r,:,0] = cifTrialsTimes[r]
    # end patch
#     indPointsLocsAndAllTimesKMS.setTimes(times=cifTrialsTimesMatrix)
#     qKAllTimes.buildKernelsMatrices()
    C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, CFilename=simInitConfig["embedding_params"]["C_filename"], dFilename=simInitConfig["embedding_params"]["d_filename"])
    print("Computing latents samples")
    latentsMeansFuncs = utils.svGPFA.configUtils.getLatentsMeansFuncs(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
#     sampledLatentsMeans = qKAllTimes.computeMeans(times=cifTrialsTimesMatrix)
#     latentsSamples, latentsMeans, latentsVariances = qKAllTimes.sample(times=cifTrialsTimesMatrix, regFactor=latentsGPRegularizationEpsilon)
#     latentsSTDs = [latentsVariances[k].sqrt() for k in range(nLatents)]
    latentsSamples, latentsMeans, latentsSTDs = utils.svGPFA.miscUtils.getLatentsSamplesMeansAndSTDs(
        nTrials=nTrials,
        meansFuncs=latentsMeansFuncs,
        kernels=kernels,
        trialsTimes=cifTrialsTimes,
        latentsGPRegularizationEpsilon=latentsGPRegularizationEpsilon,
        dtype=C.dtype)
#     latentsSamples, latentsMeans, latentsSTDs = utils.svGPFA.miscUtils.getLatentsSamplesMeansAndSTDsFromSampledMeans(
#         nTrials=nTrials,
#         sampledMeans=sampledLatentsMeans,
#         kernels=kernels,
#         trialsTimes=cifTrialsTimes,
#         latentsGPRegularizationEpsilon=latentsGPRegularizationEpsilon,
#         dtype=C.dtype)
    simulator = simulations.svGPFA.simulations.GPFASimulator()
    cifValues = simulator.getCIF(nTrials=nTrials, latentsSamples=latentsSamples, C=C, d=d, linkFunction=torch.exp)

    plt.figure()

    for k in range(nLatents):
        plt.subplot(nLatents,1,k+1)
        plt.plot(cifTrialsTimes[latentTrialToPlot], latentsSamples[latentTrialToPlot][k])
        plt.xlabel("Time (sec)")
        plt.ylabel("Latent")
        if k==0:
            plt.title("Trial: {:d}".format(latentTrialToPlot))

    plt.figure()

    plt.plot(cifTrialsTimes[cifTrialToPlot], cifValues[cifTrialToPlot][cifNeuronToPlot])
    plt.xlabel("Time (sec)")
    plt.ylabel("CIF")
    plt.title("Trial: {:d}, Neuron: {:d}".format(cifTrialToPlot, cifNeuronToPlot))

    plt.show()

    print("Getting spikes times")
    spikesTimes = simulator.simulate(cifTrialsTimes=cifTrialsTimes, cifValues=cifValues)

    # latentsMeans = qKAllTimes.computeMeans(times=cifTrialsTimesMatrix)
    # pdb.set_trace()
    # latentsSTDs = utils.svGPFA.miscUtils.getLatentsSTDs(kernels=kernels, trialsTimes=cifTrialsTimes)
    # latentsMeans, latentsSTDs = utils.svGPFA.miscUtils.getLatentsMeansAndSTDs(meansFuncs=latentsMeansFuncs, kernels=kernels, trialsTimes=cifTrialsTimes)

    spikesRates = utils.svGPFA.miscUtils.computeSpikeRates(trialsTimes=cifTrialsTimes, spikesTimes=spikesTimes)
    simRes = {"times": cifTrialsTimes, "latents": latentsSamples, "latentsMeans": latentsMeans, "latentsSTDs": latentsSTDs, "C": C, "d": d, "cifValues": cifValues, "spikes": spikesTimes}
    with open(simResFilename, "wb") as f: pickle.dump(simRes, f)

    simResConfig = configparser.ConfigParser()
    simResConfig["simulation_params"] = {"simInitConfigFilename": simInitConfigFilename}
    simResConfig["simulation_results"] = {"simResFilename": simResFilename}
    with open(simResConfigFilename, "w") as f:
        simResConfig.write(f)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
