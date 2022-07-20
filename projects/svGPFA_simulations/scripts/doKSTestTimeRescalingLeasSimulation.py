import sys
import os
import pdb
import math
import torch
import pickle
from scipy.io import loadmat
import random
import configparser
import matplotlib.pyplot as plt
sys.path.append("../src")
import plot.svGPFA.plotUtils
from stats.pointProcess.tests import KSTestTimeRescalingNumericalCorrection

def main(argv):
    gamma = 10
    trialToPlot = 0
    neuronToPlot = 2
    shuffle = False
    # estimationPrefix = 50096978
    estimationPrefix = 37816127
    ksTestTimeRescalingFigFilenamePattern = "/tmp/ksTestTR_shuffle{:d}_trial{:03d}_neuron{:03d}.png"
    deviceName = "cpu"
    initDataFilename = os.path.join(os.path.dirname(__file__), "data/pointProcessInitialConditions.mat")
    ppSimulationFilename = os.path.join(os.path.dirname(__file__), "data/pointProcessSimulation.mat")
    modelSaveFilename = \
        "results/{:08d}_leasSimulation_estimatedModel_{:s}.pickle".format(estimationPrefix, deviceName)

    mat = loadmat(initDataFilename)
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    T = math.ceil(legQuadPoints.max())

    yMat = loadmat(ppSimulationFilename)
    YNonStacked_tmp = yMat['Y']
    nTrials = YNonStacked_tmp.shape[0]
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor)

    with open(modelSaveFilename, "rb") as f: res = pickle.load(f)
    model = res["model"]

    # KS test time rescaling
    spikesTimes = YNonStacked
    dtCIF = 1e-3
    oneTrialCIFTimes = torch.arange(0, T, dtCIF)
    cifTimes = torch.unsqueeze(torch.ger(torch.ones(nTrials), oneTrialCIFTimes), dim=2)
    with torch.no_grad():
        # cifs = model.sampleCIFs(times=cifTimes)
        cifs = model.computeMeanCIFs(times=cifTimes)

    print("Processing trial {:03d}, neuron {:03d}".format(trialToPlot, neuronToPlot))
    spikesTimesKS = spikesTimes[trialToPlot][neuronToPlot]
    if shuffle:
        spikesTimesKS = [random.uniform(0, T) for i in range(len(spikesTimesKS))]
    cifTimesKS = cifTimes[trialToPlot,:,0]
    cifKS = cifs[trialToPlot][neuronToPlot]
    diffECDFsX, diffECDFsY, cb = KSTestTimeRescalingNumericalCorrection(spikesTimes=spikesTimesKS, cifTimes=cifTimesKS, cifValues=cifKS, gamma=gamma)
    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(trialToPlot, neuronToPlot, len(spikesTimesKS))
    plot.svGPFA.plotUtils.plotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, cb=cb, figFilename=ksTestTimeRescalingFigFilenamePattern.format(shuffle, trialToPlot, neuronToPlot), title=title)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
