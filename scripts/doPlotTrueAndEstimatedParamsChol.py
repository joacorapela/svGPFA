
import sys
import os
import pdb
import pickle
import argparse
import configparser
import numpy as np
import torch
import plotly.graph_objects as go
import plotly
import plotly.tools as tls
import plotly.io as pio
sys.path.append("../src")
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import plot.svGPFA.plotUtilsPlotly

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="Estimation result number", type=int)
    parser.add_argument("--latentToPlot", help="Latent to plot", type=int, default=0)
    parser.add_argument("--trialToPlot", help="Trial to plot", type=int, default=0)
    parser.add_argument("--neuronToPlot", help="Neuron to plot", type=int, default=0)
    args = parser.parse_args()

    estResNumber = args.estResNumber
    latentToPlot = args.latentToPlot
    trialToPlot = args.trialToPlot
    neuronToPlot = args.neuronToPlot

    estMetaDataFilename = "results/{:08d}_estimatedModelMetaData.ini".format(estResNumber)
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)
    latentsFigFilenamePattern = "figures/{:08d}_trueAndEstimatedLatents_latent{:03d}_trial_{:03d}.{{:s}}".format(estResNumber, latentToPlot, trialToPlot)
    indPointsMeanFigFilenamePattern = "figures/{:08d}_trueAndEstimatedIndPointsMeans_latent{:03d}_trial_{:03d}.{{:s}}".format(estResNumber, latentToPlot, trialToPlot)
    indPointsCovFigFilenamePattern = "figures/{:08d}_trueAndEstimatedIndPointsCovs_latent{:03d}_trial_{:03d}.{{:s}}".format(estResNumber, latentToPlot, trialToPlot)
    indPointsLocsFigFilenamePattern = "figures/{:08d}_trueAndEstimatedIndPointsLocs_latent{:03d}_trial_{:03d}.{{:s}}".format(estResNumber, latentToPlot, trialToPlot)
    kernelsParamsFigFilenamePattern = "figures/{:08d}_trueAndEstimatedKernelsParams.{{:s}}".format(estResNumber)
    embeddingParamsFigFilenamePattern = "figures/{:08d}_trueAndEstimatedEmbeddingParams.{{:s}}".format(estResNumber)

    estMetaDataConfig = configparser.ConfigParser()
    estMetaDataConfig.read(estMetaDataFilename)
    simResNumber = int(estMetaDataConfig["simulation_params"]["simResNumber"])
    simMetaDataFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResFilename = "results/{:08d}_simRes.pickle".format(simResNumber)
    simMetaDataConfig = configparser.ConfigParser()
    simMetaDataConfig.read(simMetaDataFilename)
    simInitConfigFilename = simMetaDataConfig["simulation_params"]["simInitConfigFilename"]
    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nLatents = int(simInitConfig["control_variables"]["nLatents"])
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    dtSimulate = float(simInitConfig["control_variables"]["dtCIF"])
    tIndPointsMeans = utils.svGPFA.configUtils.getIndPointsMeans(nTrials=nTrials, nLatents=nLatents, config=simInitConfig)

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    tKzzChol = simRes["KzzChol"]
    tIndPointsCovs = [torch.matmul(tKzzChol[k], torch.transpose(tKzzChol[k], 1, 2)) for k in range(nLatents)]
    tIndPointsLocs = simRes["indPointsLocs"]
    tTimes = simRes["times"]
    tLatentsSamples = simRes["latents"]
    tLatentsMeans = simRes["latentsMeans"]
    tLatentsSTDs = simRes["latentsSTDs"]

    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=True)
    # latentsMeansFuncs[r][k] \in lambda(t)
#     tLatentsMeansFuncs = utils.svGPFA.configUtils.getLatentsMeansFuncs(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
    CFilename = simInitConfig["embedding_params"]["C_filename"]
    dFilename = simInitConfig["embedding_params"]["d_filename"]
    trueC, trueD = utils.svGPFA.configUtils.getLinearEmbeddingParams(CFilename=CFilename, dFilename=dFilename)
    trialsTimes = utils.svGPFA.miscUtils.getTrialsTimes(trialsLengths=trialsLengths, dt=dtSimulate)

    with open(modelSaveFilename, "rb") as f: savedResults = pickle.load(f)
    model = savedResults["model"]
    kernelsParams = model.getKernelsParams()
    eTimes = tTimes
    with torch.no_grad():
        eLatentsMeans, eLatentsSTDs = model.predictLatents(newTimes=eTimes[0])
    estimatedC, estimatedD = model.getSVEmbeddingParams()
    eIndPointsLocs = model.getIndPointsLocs()

    pio.renderers.default = "browser"

    tLatentsSamplesToPlot = tLatentsSamples[trialToPlot][latentToPlot,:]
    tLatentsMeansToPlot = tLatentsMeans[trialToPlot][latentToPlot,:]
    tLatentsSTDsToPlot = tLatentsSTDs[trialToPlot][latentToPlot,:]
    eLatentsMeansToPlot = eLatentsMeans[trialToPlot,:,latentToPlot]
    eLatentsSTDsToPlot = eLatentsSTDs[trialToPlot,:,latentToPlot]
    title = "Trial {:d}, Latent {:d}".format(trialToPlot, latentToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedLatentsOneTrialOneLatent(
        tTimes = tTimes[0],
        tLatentsSamples=tLatentsSamplesToPlot,
        tLatentsMeans=tLatentsMeansToPlot,
        tLatentsSTDs=tLatentsSTDsToPlot,
        eTimes=eTimes[0],
        eLatentsMeans=eLatentsMeansToPlot,
        eLatentsSTDs=eLatentsSTDsToPlot,
        title=title,
    )
    fig.write_image(latentsFigFilenamePattern.format("png"))
    fig.write_html(latentsFigFilenamePattern.format("html"))
    fig.show()

    tKernelToPlot = kernels[latentToPlot]
    eKernelParamsToPlot = kernelsParams[latentToPlot]
    title = "Latent {:d}".format(latentToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedKernelsParamsOneLatent(trueKernel=tKernelToPlot, estimatedKernelParams=eKernelParamsToPlot, title=title)
    fig.write_image(kernelsParamsFigFilenamePattern.format("png"))
    fig.write_html(kernelsParamsFigFilenamePattern.format("html"))
    fig.show()

    svPosteriorOnIndPointsParams = model.getSVPosteriorOnIndPointsParams()

    eIndPointsMeans = svPosteriorOnIndPointsParams[:nLatents]
    tIndPointsMeansToPlot = tIndPointsMeans[trialToPlot][latentToPlot][:,0]
    eIndPointsMeansToPlot = eIndPointsMeans[latentToPlot][trialToPlot,:,0]
    srQSigmaVecs = svPosteriorOnIndPointsParams[nLatents:]
    eIndPointsCovs = utils.svGPFA.miscUtils.buildQSigmasFromSRQSigmaVecs(srQSigmaVecs=srQSigmaVecs)
    tIndPointsCovToPlot = tIndPointsCovs[latentToPlot][trialToPlot,:,:]
    tIndPointsSTDsToPlot = torch.diag(tIndPointsCovToPlot)
    eIndPointsCovToPlot = eIndPointsCovs[latentToPlot][trialToPlot,:,:]
    eIndPointsSTDsToPlot = torch.diag(eIndPointsCovToPlot)
    title = "Trial {:d}, Latent {:d}".format(trialToPlot, latentToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedIndPointsMeansOneTrialOneLatent(trueIndPointsMeans=tIndPointsMeansToPlot, estimatedIndPointsMeans=eIndPointsMeansToPlot, trueIndPointsSTDs=tIndPointsSTDsToPlot, estimatedIndPointsSTDs=eIndPointsSTDsToPlot, title=title,)
    fig.write_image(indPointsMeanFigFilenamePattern.format("png"))
    fig.write_html(indPointsMeanFigFilenamePattern.format("html"))
    fig.show()

    title = "Trial {:d}, Latent {:d}".format(trialToPlot, latentToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedIndPointsCovsOneTrialOneLatent(
        trueIndPointsCov=tIndPointsCovToPlot,
        estimatedIndPointsCov=eIndPointsCovToPlot,
        title=title,
    )
    fig.write_image(indPointsCovFigFilenamePattern.format("png"))
    fig.write_html(indPointsCovFigFilenamePattern.format("html"))
    fig.show()

#     fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedEmbeddingParams(trueC=trueC, trueD=trueD, estimatedC=estimatedC, estimatedD=estimatedD)
#     fig.write_image(embeddingParamsFigFilenamePattern.format("png"))
#     fig.write_html(embeddingParamsFigFilenamePattern.format("html"))
#     fig.show()

    tIndPointsLocsToPlot = tIndPointsLocs[latentToPlot][trialToPlot,:,0]
    eIndPointsLocsToPlot = eIndPointsLocs[latentToPlot][trialToPlot,:,0]
    title = "Trial {:d}, Latent {:d}".format(trialToPlot, latentToPlot)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedIndPointsLocsOneTrialOneLatent(trueIndPointsLocs=tIndPointsLocsToPlot, estimatedIndPointsLocs=eIndPointsLocsToPlot, title=title)
    fig.write_image(indPointsLocsFigFilenamePattern.format("png"))
    fig.write_html(indPointsLocsFigFilenamePattern.format("html"))
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
