
import sys
import os
import pdb
import pickle
import configparser
import numpy as np
import torch
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import plotly.tools as tls
sys.path.append("../src")
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils

def plotTrueAndEstimatedEmbeddingParams(trueC, trueD, estimatedC, estimatedD,
                                        staticFigFilename,
                                        dynamicFigFilename,
                                        linestyleTrue="solid",
                                        linestyleEstimated="dash",
                                        xlabel="Neuron Index",
                                        ylabel="Coefficient Value"):
    neuronIndices = np.arange(trueC.shape[0])
    fig = go.Figure()
    for i in range(trueC.shape[1]):
        fig.add_trace(go.Scatter(
            x=neuronIndices,
            y=trueC[:,i],
            mode="lines+markers",
            showlegend=True,
            name="true C[{:d}]".format(i),
            line=dict(dash=linestyleTrue),
        ))
        fig.add_trace(go.Scatter(
            x=neuronIndices,
            y=estimatedC[:,i],
            mode="lines+markers",
            showlegend=True,
            name="estimated C[{:d}]".format(i),
            line=dict(dash=linestyleEstimated),
        ))
    fig.add_trace(go.Scatter(
        x=neuronIndices,
        y=trueD[:,0],
        mode="lines+markers",
        showlegend=True,
        name="true d",
        line=dict(dash=linestyleTrue),
    ))
    fig.add_trace(go.Scatter(
        x=neuronIndices,
        y=estimatedD[:,0],
        mode="lines+markers",
        showlegend=True,
        name="estimated d",
        line=dict(dash=linestyleEstimated),
    ))
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel)
    fig.write_image(staticFigFilename)
    plotly.offline.plot(fig, filename=dynamicFigFilename)

def plotTrueAndEstimatedLatentsMeans(trueLatentsMeans, estimatedLatentsMeans,
                                     trialsTimes, 
                                     staticFigFilename,
                                     dynamicFigFilename,
                                     colorTrue="blue",
                                     colorEstimated="red",
                                     labelTrue="True",
                                     labelEstimated="Estimated",
                                     xlabel="Time (sec)",
                                     ylabel="Latent Value"):
    def getTracesOneSetTrueAndEstimatedLatentsMeans(
        trueLatentMean,
        estimatedLatentMean,
        times,
        labelTrue, labelEstimated,
        useLegend):
        traceTrue = go.Scatter(
            x=times,
            y=trueLatentMean,
            mode="lines+markers",
            name=labelTrue,
            line=dict(color=colorTrue),
            showlegend=useLegend)
        traceEstimated = go.Scatter(
            x=times,
            y=estimatedLatentMean,
            mode="lines+markers",
            name=labelEstimated,
            line=dict(color=colorEstimated),
            showlegend=useLegend)
        return traceTrue, traceEstimated

    # trueLatentsMeans[r] \in nLatents x nInd[k]
    # qMu[k] \in nTrials x nInd[k] x 1
    nTrials = len(trueLatentsMeans)
    nLatents = trueLatentsMeans[0].shape[0]
    fig = make_subplots(rows=nTrials, cols=nLatents)
    for r in range(nTrials):
        times = trialsTimes[r]
        for k in range(nLatents):
            trueLatentMean = trueLatentsMeans[r][k,:]
            estimatedLatentMean = estimatedLatentsMeans[r,:,k]
            if r==0 and k==nLatents-1:
                useLegend = True
            else:
                useLegend = False
            traceTrue, traceEstimated = getTracesOneSetTrueAndEstimatedLatentsMeans(
                trueLatentMean=trueLatentMean,
                estimatedLatentMean=estimatedLatentMean,
                times=times,
                labelTrue=labelTrue,
                labelEstimated=labelEstimated,
                useLegend=useLegend)
            fig.add_trace(traceTrue, row=r+1, col=k+1)
            fig.add_trace(traceEstimated, row=r+1, col=k+1)
    fig.update_yaxes(title_text=ylabel, row=nTrials//2+1, col=1)
    fig.update_xaxes(title_text=xlabel, row=nTrials, col=nLatents//2+1)
    fig.write_image(staticFigFilename)
    plotly.offline.plot(fig, filename=dynamicFigFilename)

def plotTrueAndEstimatedKernelsParams(trueKernels, estimatedKernelsParams,
                                      staticFigFilename, dynamicFigFilename,
                                      colorTrue="blue",
                                      colorEstimated="red",
                                      trueLegend="True",
                                      estimatedLegend="Estimated"):
    nLatents = len(trueKernels)
    titles = ["Kernel {:d}: {:s}".format(i, trueKernels[i].__class__.__name__) for i in range(nLatents)]
    fig = tls.make_subplots(rows=nLatents, cols=1, subplot_titles=titles)
    for k in range(nLatents):
        namedParams = trueKernels[k].getNamedParams()
        labels = list(namedParams.keys())
        trueParams = [z.item() for z in list(namedParams.values())]
        estimatedParams = estimatedKernelsParams[k].tolist()
        # we are fixing scale to 1.0. This is not great :(
        estimatedParams = [1.0] + estimatedParams
        if k==0:
            showLegend = True
        else:
            showLegend = False
        traceTrue = go.Bar(x=labels, y=trueParams, name=trueLegend, marker_color=colorTrue, showlegend=showLegend)
        traceEstimated = go.Bar(x=labels, y=estimatedParams, name=estimatedLegend, marker_color=colorEstimated, showlegend=showLegend)
        fig.append_trace(traceTrue, k+1, 1)
        fig.append_trace(traceEstimated, k+1, 1)
    fig.update_yaxes(title_text="Parameter Value", row=nLatents//2+1, col=1)
    fig.write_image(staticFigFilename)
    plotly.offline.plot(fig, filename=dynamicFigFilename)

def main(argv):
    if len(argv)!=2:
        print("Usage {:s} <estimation number> ".format(argv[0]))
        return

    estNumber = int(argv[1])
    gpRegularization = 1e-3
    estMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(estNumber)
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estNumber)
    kernelsParamsFigFilenamePattern = "figures/{:08d}_trueAndEstimatedKernelsParams.{{:s}}".format(estNumber)
    latentsMeansFigFilenamePattern = "figures/{:08d}_trueAndEstimatedLatentsMeans.{{:s}}".format(estNumber)
    embeddingParamsFigFilenamePattern = "figures/{:08d}_trueAndEstimatedEmbeddingParams.{{:s}}".format(estNumber)

    estMetaDataConfig = configparser.ConfigParser()
    estMetaDataConfig.read(estMetaDataFilename)
    simResNumber = int(estMetaDataConfig["simulation_params"]["simResNumber"])
    simMetaDataFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simMetaDataConfig = configparser.ConfigParser()
    simMetaDataConfig.read(simMetaDataFilename)
    simInitConfigFilename = simMetaDataConfig["simulation_params"]["simInitConfigFilename"]
    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nLatents = int(simInitConfig["control_variables"]["nLatents"])
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    dtSimulate = float(simInitConfig["control_variables"]["dt"])

    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)[0]
    # latentsMeansFuncs[r][k] \in lambda(t)
    tLatentsMeansFuncs = utils.svGPFA.configUtils.getLatentsMeansFuncs(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
    CFilename = simInitConfig["embedding_params"]["C_filename"]
    dFilename = simInitConfig["embedding_params"]["d_filename"]
    trueC, trueD = utils.svGPFA.configUtils.getLinearEmbeddingParams(nNeurons=nNeurons, nLatents=nLatents, CFilename=CFilename, dFilename=dFilename)
    trialsTimes = utils.svGPFA.miscUtils.getTrialsTimes(trialsLengths=trialsLengths, dt=dtSimulate)

    # latentsMeansSamples[r][k,t]
    tLatentsMeans = utils.svGPFA.miscUtils.getLatentsMeanFuncsSamples(latentsMeansFuncs=
                                                tLatentsMeansFuncs,
                                               trialsTimes=trialsTimes,
                                               dtype=trueC.dtype)

    with open(modelSaveFilename, "rb") as f: savedResults = pickle.load(f)
    model = savedResults["model"]
    kernelsParams = model.getKernelsParams()
    with torch.no_grad():
        latentsMeans, _ = model.predictLatents(newTimes=trialsTimes[0])
        estimatedC, estimatedD = model.getSVEmbeddingParams()
        plotTrueAndEstimatedKernelsParams(trueKernels=kernels, 
                                          estimatedKernelsParams=kernelsParams,
                                          staticFigFilename=kernelsParamsFigFilenamePattern.format("png"),
                                          dynamicFigFilename=kernelsParamsFigFilenamePattern.format("html"))

        # qMu[r] \in nTrials x nInd[k] x 1
        plotTrueAndEstimatedLatentsMeans(trueLatentsMeans=tLatentsMeans,
                                         estimatedLatentsMeans=latentsMeans,
                                         trialsTimes=trialsTimes, 
                                         staticFigFilename=latentsMeansFigFilenamePattern.format("png"),
                                         dynamicFigFilename=latentsMeansFigFilenamePattern.format("html"))

        plotTrueAndEstimatedEmbeddingParams(trueC=trueC, trueD=trueD,
                                            estimatedC=estimatedC,
                                            estimatedD=estimatedD,
                                            staticFigFilename=embeddingParamsFigFilenamePattern.format("png"),
                                            dynamicFigFilename=embeddingParamsFigFilenamePattern.format("html"))

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
