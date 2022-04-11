import sys
import os
import pdb
import pickle
sys.path.append("../src")
import plot.svGPFA.plotUtils

def main(argv):
    if len(argv)!=4:
        print("Usage {:s} <simulation result number> <trial> <neuron>".format(argv[0]))
        return

    simResNumber = int(argv[1])
    trialToPlot = int(argv[2])
    neuronToPlot = int(argv[3])
    dtCIF = 1e-3

    simResFilename = "results/{:08d}_simRes.pickle".format(simResNumber)
    cifFigFilename = "figures/{:08d}_simulation_cif_trial{:03d}_neuron{:03d}.png".format(simResNumber, trialToPlot, neuronToPlot)

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    cifTimes = simRes["cifTimes"]
    cifValues = simRes["cifValues"]

    timesCIFToPlot = cifTimes[trialToPlot]
    valuesCIFToPlot = cifValues[trialToPlot][neuronToPlot]
    title = "Trial {:d}, Neuron {:d}".format(trialToPlot, neuronToPlot)
    plot.svGPFA.plotUtils.plotCIF(times=timesCIFToPlot, values=valuesCIFToPlot, title=title, figFilename=cifFigFilename)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
