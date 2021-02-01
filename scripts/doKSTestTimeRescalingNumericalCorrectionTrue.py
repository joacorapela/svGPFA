import sys
import os
import pdb
import math
import torch
import pickle
sys.path.append("../src")
import plot.svGPFA.plotUtils
from stats.pointProcess.tests import KSTestTimeRescalingNumericalCorrection
def main(argv):
    if len(argv)!=5:
        print("Usage {:s} <simulation result number> <number of KS test resamples> <trial> <neuron>".format(argv[0]))
        return

    simResNumber = int(argv[1])
    gamma = int(argv[2])
    trialToAnalyze = int(argv[3])
    neuronToAnalyze = int(argv[4])
    dtCIF = 1e-3

    simResFilename = "results/{:08d}_simRes.pickle".format(simResNumber)
    figFilename = "figures/{:08d}_simulation_ksTestTimeRescaling_trial{:03d}_neuron{:03d}.png".format(simResNumber, trialToAnalyze, neuronToAnalyze)

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    cifTimes = simRes["cifTimes"]
    cifValues = simRes["cifValues"]

    spikesTimesKS = spikesTimes[trialToAnalyze][neuronToAnalyze]
    cifTimesKS = cifTimes[trialToAnalyze]
    cifValuesKS = cifValues[trialToAnalyze][neuronToAnalyze]

    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = KSTestTimeRescalingNumericalCorrection(spikesTimes=spikesTimesKS, cifTimes=cifTimesKS, cifValues=cifValuesKS, gamma=gamma)
    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(trialToAnalyze, neuronToAnalyze, len(spikesTimesKS))
    plot.svGPFA.plotUtils.plotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx, estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb, figFilename=figFilename, title=title)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
