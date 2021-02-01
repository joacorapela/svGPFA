import sys
import os
import pdb
import math
import torch
import pickle
import configparser
import pandas as pd
import statsmodels.tsa.stattools
import sklearn.metrics
sys.path.append("../src")
import plot.svGPFA.plotUtils

def main(argv):
    if len(argv)!=4:
        print("Usage {:s} <estimation result number> <trial> <neuron>".format(argv[0]))
        return

    estResNumber = int(argv[1])
    trialToAnalyze = int(argv[2])
    neuronToAnalyze = int(argv[3])
    dtCIF = 1e-3

    rocFigFilename = "figures/{:08d}_rocAnalysis_trial{:03d}_neuron{:03d}.png".format(estResNumber, trialToAnalyze, neuronToAnalyze)

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

    T = torch.tensor(trialsLengths).max().item()
    oneTrialCIFTimes = torch.arange(0, T, dtCIF)
    cifTimes = torch.unsqueeze(torch.ger(torch.ones(nTrials), oneTrialCIFTimes), dim=2)
    with torch.no_grad():
        # cifs = model.sampleCIFs(times=cifTimes)
        cifValues = model.computeMeanCIFs(times=cifTimes)
    spikesTimesKS = spikesTimes[trialToAnalyze][neuronToAnalyze]
    cifTimesKS = cifTimes[trialToAnalyze,:,0]
    cifValuesKS = cifValues[trialToAnalyze][neuronToAnalyze]

    pk = cifValuesKS*dtCIF
    bins = pd.interval_range(start=0, end=T, periods=len(pk))
    # start binning spikes using pandas
    cutRes, _ = pd.cut(spikesTimesKS, bins=bins, retbins=True)
    Y = torch.from_numpy(cutRes.value_counts().values)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, pk, pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(trialToAnalyze, neuronToAnalyze, len(spikesTimesKS))
    plot.svGPFA.plotUtils.plotResROCAnalysis(fpr=fpr, tpr=tpr, auc=roc_auc, title=title, figFilename=rocFigFilename)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
