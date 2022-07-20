
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
    parser.add_argument("--cifNeuronToPlot", help="Neuron on which to plot the CIF", type=int, default=0)
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
    indPointsLocsKMSRegEpsilon = float(simInitConfig["control_variables"]["indPointsLocsKMSRegEpsilon"])
    latentsCovRegEpsilon = float(simInitConfig["control_variables"]["latentsCovRegEpsilon"])

    randomPrefixUsed = True
    while randomPrefixUsed:
        simNumber = random.randint(0, 10**8)
        simResConfigFilename = \
            "results/{:08d}_simulation_metaData.ini".format(simNumber)
        if not os.path.exists(simResConfigFilename):
           randomPrefixUsed = False
    simResFilename = "results/{:08d}_simRes.pickle".format(simNumber)

    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=False)
    indPointsLocs = utils.svGPFA.configUtils.getIndPointsLocs0(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
    latentsTrialsTimes = utils.svGPFA.miscUtils.getTrialsTimes(trialsLengths=trialsLengths, dt=dtCIF)
    latentsTrialsTimesMatrix = torch.empty((nTrials, len(latentsTrialsTimes[0]), 1))
    # patch to accomodate unreasonable need to have trial times of equal length
    # across trials
    for r in range(nTrials):
        latentsTrialsTimesMatrix[r,:,0] = latentsTrialsTimes[r]
    # end patch
    C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(CFilename=simInitConfig["embedding_params"]["C_filename"], dFilename=simInitConfig["embedding_params"]["d_filename"])
    indPointsMeans = utils.svGPFA.configUtils.getIndPointsMeans(nTrials=nTrials, nLatents=nLatents, config=simInitConfig)
    simulator = simulations.svGPFA.simulations.GPFAwithIndPointsSimulator()
    print("Computing latents samples")
    latentsSamples, latentsMeans, latentsSTDs, Kzz = simulator.getLatentsSamplesMeansAndSTDs(
        indPointsMeans=indPointsMeans,
        kernels=kernels,
        indPointsLocs=indPointsLocs,
        trialsTimes=latentsTrialsTimesMatrix,
        indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon,
        latentsCovRegEpsilon=latentsCovRegEpsilon,
        dtype=C.dtype,
    )
#     exit = False
#     attemptNumber = 1
#     while not exit:
#         latentsSamples, latentsMeans, latentsSTDs, KzzChol = simulator.getLatentsSamplesMeansAndSTDs(
#             indPointsMeans=indPointsMeans,
#             kernels=kernels,
#             indPointsLocs=indPointsLocs,
#             trialsTimes=latentsTrialsTimesMatrix,
#             indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon,
#             latentsCovRegEpsilon=latentsCovRegEpsilon,
#             dtype=C.dtype)
#         maxLatentsSamples = torch.tensor([torch.max(latentsSamples[k]) for k in range(nLatents)]).max()
#         print("Attempt number: {:d}, max lLatents: {:.02f}".format(attemptNumber, maxLatentsSamples))
# 
#         if(maxLatentsSamples>1.0):
#             exit = True
#         else:
#             attemptNumber += 1
    cifValues = simulator.getCIF(nTrials=nTrials, latentsSamples=latentsSamples, C=C, d=d, linkFunction=torch.exp)

    plt.figure()

    for k in range(nLatents):
        plt.subplot(nLatents,1,k+1)
        plt.plot(latentsTrialsTimes[latentTrialToPlot], latentsSamples[latentTrialToPlot][k])
        plt.xlabel("Time (sec)")
        plt.ylabel("Latent")
        if k==0:
            plt.title("Trial: {:d}".format(latentTrialToPlot))

    plt.figure()
    plt.show(block=False)

    for n in range(nNeurons):
        plt.plot(latentsTrialsTimes[cifTrialToPlot], cifValues[cifTrialToPlot][n])
    plt.xlabel("Time (sec)")
    plt.ylabel("CIF")
    plt.title("Trial: {:d}".format(cifTrialToPlot))

    plt.show()
    pdb.set_trace()

    print("Getting spikes times")
    sampling_func = stats.pointProcess.sampling.sampleInhomogeneousPP_thinning
    spikesTimes = simulator.simulate(cifTrialsTimes=latentsTrialsTimes,
                                     cifValues=cifValues,
                                     sampling_func=sampling_func)

    spikesRates = utils.svGPFA.miscUtils.computeSpikeRates(trialsTimes=latentsTrialsTimes, spikesTimes=spikesTimes)
    simRes = {"latentsTrialsTimes": latentsTrialsTimes,
              "latentsSamples": latentsSamples,
              "latentsMeans": latentsMeans,
              "latentsSTDs": latentsSTDs,
              "indPointsMeans": indPointsMeans,
              "indPointsLocs": indPointsLocs,
              "C": C, "d": d,
              "cifValues": cifValues,
              "spikes": spikesTimes,
              "Kzz": Kzz}
    with open(simResFilename, "wb") as f: pickle.dump(simRes, f)

    simResConfig = configparser.ConfigParser()
    simResConfig["simulation_params"] = {"simInitConfigFilename": simInitConfigFilename}
    simResConfig["simulation_results"] = {"simResFilename": simResFilename}
    with open(simResConfigFilename, "w") as f:
        simResConfig.write(f)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
