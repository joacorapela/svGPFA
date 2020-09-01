
import sys
import os
import torch
import pdb
import pickle
import argparse
import configparser
from scipy.io import loadmat
sys.path.append(os.path.expanduser("../src"))
import plot.svGPFA.plotUtilsPlotly
import utils.svGPFA.configUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--trialToPlot", help="Trial to plot", type=int, default=0)
    parser.add_argument("--neuronToPlot", help="Trial to plot", type=int, default=0)
    args = parser.parse_args()
    estResNumber = args.estResNumber
    trialToPlot = args.trialToPlot
    neuronToPlot = args.neuronToPlot

    estimResMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(estResNumber)
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)
    figFilenamePattern = "figures/{:08d}_trueAndEstimatedEmbedding_trial{:d}_neuron{:d}.{{:s}}".format(estResNumber, trialToPlot, neuronToPlot)

    estimResConfig = configparser.ConfigParser()
    estimResConfig.read(estimResMetaDataFilename)
    simResNumber = int(estimResConfig["simulation_params"]["simResNumber"])
    simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]
    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    CFilename = simInitConfig["embedding_params"]["C_filename"]
    dFilename = simInitConfig["embedding_params"]["d_filename"]
    tC, td = utils.svGPFA.configUtils.getLinearEmbeddingParams(CFilename=CFilename, dFilename=dFilename)
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    simResFilename = simResConfig["simulation_results"]["simResFilename"]

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    tTimes = simRes["times"]
    # tLatentsSamples[r], tLatentsMeans[r], tLatentsSTDs[r] \in nLatents x nSamples
    tLatentsSamples = simRes["latents"]
    tLatentsMeans = simRes["latentsMeans"]
    tLatentsSTDs = simRes["latentsSTDs"]

    with open(modelSaveFilename, "rb") as f: estResults = pickle.load(f)
    model = estResults["model"]
    eC, ed = model.getSVEmbeddingParams()
    with torch.no_grad():
        # testMuK, testVarK \in nTrial x nSamples x nLatents
        muK, varK = model.predictLatents(newTimes=tTimes[0])
    eTimes = tTimes
    # eLatentsMeans[r], eLatentsSTDs[r] \in nLatents x nSamples
    eLatentsMeans = [muK[r,:,:].transpose(0, 1) for r in range(nTrials)]
    eLatentsSTDs = [varK[r,:,:].sqrt().transpose(0, 1) for r in range(nTrials)]

    # tEmbeddingSamples[r], tEmbeddingMeans[r], tEmbeddingSTDs \in nNeurons x nSamples
    tEmbeddingSamples = [torch.matmul(tC, tLatentsSamples[r])+td for r in range(nTrials)]
    tEmbeddingMeans = [torch.matmul(tC, tLatentsMeans[r])+td for r in range(nTrials)]
    tEmbeddingSTDs = [torch.matmul(tC, tLatentsSTDs[r]) for r in range(nTrials)]

    # eEmbeddingMeans[r], eEmbeddingSTDs \in nNeurons x nSamples
    eEmbeddingMeans = [torch.matmul(eC, eLatentsMeans[r]) for r in range(nTrials)]
    eEmbeddingSTDs = [torch.matmul(eC, eLatentsSTDs[r]) for r in range(nTrials)]

    tSamplesToPlot = tEmbeddingSamples[trialToPlot][neuronToPlot,:]
    tMeansToPlot = tEmbeddingMeans[trialToPlot][neuronToPlot,:]
    tSTDsToPlot = tEmbeddingSTDs[trialToPlot][neuronToPlot,:]
    eMeansToPlot = eEmbeddingMeans[trialToPlot][neuronToPlot,:]
    eSTDsToPlot = eEmbeddingSTDs[trialToPlot][neuronToPlot,:]
    title = "Trial {:d}, Neuron {:d}".format(trialToPlot, neuronToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedEmbedding(tTimes=tTimes[0], tSamples=tSamplesToPlot, tMeans=tMeansToPlot, tSTDs=tSTDsToPlot, eTimes=eTimes[0], eMeans=eMeansToPlot, eSTDs=eSTDsToPlot, title=title)

    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
