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
    if len(argv)!=2:
        print("Usage {:s} <estimation result number>".format(argv[0]))
        return

    # load data and initial values
    estResNumber = int(argv[1])
    ksTestTimeRescalingFigFilename = "figures/{:08d}_ksTestTimeRescaling.png".format(estResNumber)

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
    trialKSTestTimeRescaling = 0
    neuronKSTestTimeRescaling = 0
    dtCIF = 1e-3
    T = torch.tensor(trialsLengths).max()
    oneTrialCIFTimes = torch.arange(0, T, dtCIF)
    cifTimes = torch.unsqueeze(torch.ger(torch.ones(nTrials), oneTrialCIFTimes), dim=2)
    with torch.no_grad():
        cifs = model.sampleCIFs(times=cifTimes)
    spikesTimesKS = spikesTimes[trialKSTestTimeRescaling][neuronKSTestTimeRescaling]
    cifKS = cifs[trialKSTestTimeRescaling][neuronKSTestTimeRescaling]
    sUTRISIs, uCDF, cb = KSTestTimeRescalingUnbinned(spikesTimes=spikesTimesKS, cif=cifKS, t0=0, tf=T, dt=dtCIF)
    title = "Trial {:d}, Neuron {:d}".format(trialKSTestTimeRescaling, neuronKSTestTimeRescaling)
    plot.svGPFA.plotUtils.plotResKSTestTimeRescaling(sUTRISIs=sUTRISIs, uCDF=uCDF, cb=cb, figFilename=ksTestTimeRescalingFigFilename, title=title)
    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
