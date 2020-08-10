
import sys
import os
import pdb
import pickle
import configparser
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly
import plotly.tools as tls
sys.path.append("../src")
from plot.svGPFA.plotUtils import plotSimulatedAndEstimatedCIFs

def main(argv):
    if len(argv)!=4:
        print("Usage {:s} <estimation number> <trial> <neuron>".format(argv[0]))
        return

    estNumber = int(argv[1])
    trialToPlot = int(argv[2])
    neuronToPlot = int(argv[3])

    estMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(estNumber)
    modelFilename = "results/{:08d}_estimatedModel.pickle".format(estNumber)
    figFilename = "figures/{:08d}_trueAndEstimatedCIFs_trial{:03d}_neuron{:03d}.png".format(estNumber, trialToPlot, neuronToPlot)

    estMetaDataConfig = configparser.ConfigParser()
    estMetaDataConfig.read(estMetaDataFilename)
    simResNumber = int(estMetaDataConfig["simulation_params"]["simResNumber"])

    simResFilename = "results/{:08d}_simRes.pickle".format(simResNumber)
    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    # cifTimes = simRes["cifTimes"]
    cifTimes = simRes["times"]
    nTrials = len(cifTimes)
    oneTrialCIFTimes = cifTimes[trialToPlot]
    cifTimes = torch.unsqueeze(torch.ger(torch.ones(nTrials), oneTrialCIFTimes), dim=2)
    simCIFsValues = simRes["cifValues"]

    with open(modelFilename, "rb") as f: modelRes = pickle.load(f)

    model = modelRes["model"]
    estCIFsValues = model.computeMeanCIFs(times=cifTimes)

    title = "Trial {:d}, Neuron {:d}".format(trialToPlot, neuronToPlot)
    plotSimulatedAndEstimatedCIFs(times=cifTimes[trialToPlot, :, 0], simCIFValues=simCIFsValues[trialToPlot][neuronToPlot], estCIFValues=estCIFsValues[trialToPlot][neuronToPlot].detach(), figFilename=figFilename, title=title)

    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
