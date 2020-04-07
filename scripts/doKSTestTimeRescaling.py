import sys
import os
import pdb
import torch
import pickle
import configparser
sys.path.append(os.path.expanduser("../src"))
import plot.svGPFA.plotUtils
from stats.pointProcess.tests import KSTestTimeRescalingUnbinned

def main(argv):
    if len(argv)!=4:
        print("Usage {:s} <estimation result number> <trial to plot> <neuron to plot>".format(argv[0]))
        return

    estResNumber = int(argv[1])
    trialToPlot = int(argv[2])
    neuronToPlot = int(argv[3])
    dtCIF = 1e-3

    ksTestTimeRescalingFigFilename = "figures/{:08d}_ksTestTimeRescaling_trial{:03d}_neuron_{:04d}.png".format(estResNumber, trialToPlot, neuronToPlot)

    # load data and initial values
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)
    with open(modelSaveFilename, "rb") as f: savedResults = pickle.load(f)
    model = savedResults["model"]

    estResConfigFilename = "results/{:08d}_estimation_metaData.ini".format(estResNumber)
    estResConfig = configparser.ConfigParser()
    estResConfig.read(estResConfigFilename)
    simResNumber = int(estResConfig["simulation_params"]["simResNumber"])
    simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)

    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]
    simResFilename = simResConfig["simulation_results"]["simResFilename"]

    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nLatents = int(simInitConfig["control_variables"]["nLatents"])
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]

    # KS test time rescaling
    T = torch.tensor(trialsLengths).max()
    oneTrialCIFTimes = torch.arange(0, T, dtCIF)
    cifTimes = torch.unsqueeze(torch.ger(torch.ones(nTrials), oneTrialCIFTimes), dim=2)
    with torch.no_grad():
        cifs = model.sampleCIFs(times=cifTimes)
    spikesTimesKS = spikesTimes[trialToPlot][neuronToPlot]
    ### begin debug ###
    # print("*** Warning debug code on ***")
    # import random
    # spikesTimesKS = [random.uniform(0, 1) for i in range(len(spikesTimesKS))]
    ### end debug ###
    cifKS = cifs[trialToPlot][neuronToPlot]
    sUTRISIs, uCDF, cb = KSTestTimeRescalingUnbinned(spikesTimes=spikesTimesKS, cif=cifKS, t0=0, tf=T, dt=dtCIF)
    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(trialToPlot, neuronToPlot, len(spikesTimesKS))
    plot.svGPFA.plotUtils.plotResKSTestTimeRescaling(sUTRISIs=sUTRISIs, uCDF=uCDF, cb=cb, figFilename=ksTestTimeRescalingFigFilename, title=title)
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
