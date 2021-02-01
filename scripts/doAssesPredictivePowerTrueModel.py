import sys
import os
import pdb
import math
import torch
import pickle
import pandas as pd
from sklearn import metrics
sys.path.append("../src")
import plot.svGPFA.plotUtils

def main(argv):
    if len(argv)!=4:
        print("Usage {:s} <simulation result number> <trial> <neuron>".format(argv[0]))
        return

    simResNumber = int(argv[1])
    trialToAnalyze = int(argv[2])
    neuronToAnalyze = int(argv[3])
    dtCIF = 1e-3

    simResFilename = "results/{:08d}_simRes.pickle".format(simResNumber)
    rocFigFilename = "figures/{:08d}_simulation_rocAnalysis_trial{:03d}_neuron{:03d}.png".format(simResNumber, trialToAnalyze, neuronToAnalyze)

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    cifTimes = simRes["cifTimes"]
    cifValues = simRes["cifValues"]

    spikesTimesKS = spikesTimes[trialToAnalyze][neuronToAnalyze]
    cifTimesKS = cifTimes[trialToAnalyze]
    cifValuesKS = cifValues[trialToAnalyze][neuronToAnalyze]
    T = math.ceil(cifTimesKS.max())
    pk = cifValuesKS*dtCIF
    bins = pd.interval_range(start=0, end=T, periods=len(pk))
    # start binning spikes using pandas
    cutRes, _ = pd.cut(spikesTimesKS, bins=bins, retbins=True)
    Y = torch.from_numpy(cutRes.value_counts().values)

    fpr, tpr, thresholds = metrics.roc_curve(Y, pk, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(trialToAnalyze, neuronToAnalyze, len(spikesTimesKS))
    plot.svGPFA.plotUtils.plotResROCAnalysis(fpr=fpr, tpr=tpr, auc=roc_auc, title=title, figFilename=rocFigFilename)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
