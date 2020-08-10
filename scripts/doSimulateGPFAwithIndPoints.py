
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
    nIndPointsPerLatent = [int(str) for str in simInitConfig["control_variables"]["nIndPointsPerLatent"][1:-1].split(",")]
    nLatents = len(nIndPointsPerLatent)
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    T = torch.tensor(trialsLengths).max().item()
    dtCIF = float(simInitConfig["control_variables"]["dtCIF"])
    indPointsLocsKMSEpsilon = float(simInitConfig["control_variables"]["indPointsLocsKMSEpsilon"])
    firstIndPointLoc = float(simInitConfig["control_variables"]["firstIndPointLoc"])

    randomPrefixUsed = True
    while randomPrefixUsed:
        simNumber = random.randint(0, 10**8)
        simResConfigFilename = \
            "results/{:08d}_simulation_metaData.ini".format(simNumber)
        if not os.path.exists(simResConfigFilename):
           randomPrefixUsed = False
    simResFilename = "results/{:08d}_simRes.pickle".format(simNumber)

    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=False)
    indPointsLocs = utils.svGPFA.initUtils.getIndPointLocs0(nIndPointsPerLatent=nIndPointsPerLatent, trialsLengths=trialsLengths, firstIndPointLoc=firstIndPointLoc)
    cifTrialsTimes = utils.svGPFA.miscUtils.getTrialsTimes(trialsLengths=trialsLengths, dt=dtCIF)
    cifTrialsTimesMatrix = torch.empty((nTrials, len(cifTrialsTimes[0]), 1))
    # patch to accomodate unreasonable need to have trial times of equal length
    # across trials
    for r in range(nTrials):
        cifTrialsTimesMatrix[r,:,0] = cifTrialsTimes[r]
    # end patch
    C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, CFilename=simInitConfig["embedding_params"]["C_filename"], dFilename=simInitConfig["embedding_params"]["d_filename"])
    indPointsMeans = utils.svGPFA.configUtils.getIndPointsMeans(nTrials=nTrials, nIndPointsPerLatent=nIndPointsPerLatent, config=simInitConfig)
    simulator = simulations.svGPFA.simulations.GPFAwithIndPointsSimulator()
    print("Computing latents samples")
    latentsMeans, KzzChol = simulator.getLatentsMeans(
        indPointsMeans=indPointsMeans,
        kernels=kernels,
        indPointsLocs=indPointsLocs,
        trialsTimes=cifTrialsTimesMatrix,
        indPointsLocsKMSEpsilon=indPointsLocsKMSEpsilon,
        dtype=C.dtype)
    cifValues = simulator.getCIF(nTrials=nTrials, latentsMeans=latentsMeans, C=C, d=d, linkFunction=torch.exp)

    plt.figure()

    for k in range(nLatents):
        plt.subplot(nLatents,1,k+1)
        plt.plot(cifTrialsTimes[latentTrialToPlot], latentsMeans[latentTrialToPlot][k])
        plt.xlabel("Time (sec)")
        plt.ylabel("Latent Mean")
        if k==0:
            plt.title("Trial: {:d}".format(latentTrialToPlot))

    plt.figure()

    plt.plot(cifTrialsTimes[cifTrialToPlot], cifValues[cifTrialToPlot][cifNeuronToPlot])
    plt.xlabel("Time (sec)")
    plt.ylabel("CIF")
    plt.title("Trial: {:d}, Neuron: {:d}".format(cifTrialToPlot, cifNeuronToPlot))

    plt.show()
    pdb.set_trace()

    print("Getting spikes times")
    spikesTimes = simulator.simulate(cifTrialsTimes=cifTrialsTimes, cifValues=cifValues)

    spikesRates = utils.svGPFA.miscUtils.computeSpikeRates(trialsTimes=cifTrialsTimes, spikesTimes=spikesTimes)
    simRes = {"times": cifTrialsTimes, "latentsMeans": latentsMeans, "indPointsMeans": indPointsMeans, "KzzChol": KzzChol, "C": C, "d": d, "cifValues": cifValues, "spikes": spikesTimes}
    with open(simResFilename, "wb") as f: pickle.dump(simRes, f)

    simResConfig = configparser.ConfigParser()
    simResConfig["simulation_params"] = {"simInitConfigFilename": simInitConfigFilename}
    simResConfig["simulation_results"] = {"simResFilename": simResFilename}
    with open(simResConfigFilename, "w") as f:
        simResConfig.write(f)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
