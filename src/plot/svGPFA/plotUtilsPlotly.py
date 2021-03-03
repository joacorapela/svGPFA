
import pdb
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.subplots
import plotly
import plotly.io as pio
import plotly.express as px

# spike rates and times
def getPlotSpikeRatesForAllTrialsAndAllNeurons(spikesRates, xlabel="Neuron", ylabel="Average Spike Rate (Hz)", legendLabelPattern = "Trial {:d}"):
    nTrials = spikesRates.shape[0]
    nNeurons = spikesRates.shape[1]

    data = []
    layout = {
        "xaxis": {"title": xlabel},
        "yaxis": {"title": ylabel},
    }
    neuronsIndices = np.arange(nNeurons)
    for r in range(nTrials):
        data.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": legendLabelPattern.format(r),
                "x": neuronsIndices,
                "y": spikesRates[r,:]
            },
        )
    fig = go.Figure(
        data=data,
        layout=layout,
    )
    return fig

def getSimulatedSpikesTimesPlotMultipleTrials(spikesTimes, xlabel="Time (sec)", ylabel="Neuron", titlePattern="Trial {:d}"):
    nTrials = len(spikesTimes)
    subplotsTitles = ["trial={:d}".format(r) for r in range(nTrials)]
    fig = plotly.subplots.make_subplots(rows=nTrials, cols=1, shared_xaxes=True, shared_yaxes=True, subplot_titles=subplotsTitles)
    for r in range(nTrials):
        for n in range(len(spikesTimes[r])):
            trace = go.Scatter(
                x=spikesTimes[r][n].numpy(),
                y=n*np.ones(len(spikesTimes[r][n])),
                mode="markers",
                marker=dict(size=3, color="black"),
                showlegend=False,
                # hoverinfo="skip",
            )
            fig.add_trace(trace, row=r+1, col=1)
        if r==nTrials-1:
            fig.update_xaxes(title_text=xlabel, row=r+1, col=1)
        if r==math.floor(nTrials/2):
            fig.update_yaxes(title_text=ylabel, row=r+1, col=1)
    fig.update_layout(
        {
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        }
    )
    return fig

def getSimulatedSpikesTimesPlotOneTrial(spikesTimes, title, xlabel="Time (sec)", ylabel="Neuron"):
    fig = go.Figure()
    for n in range(len(spikesTimes)):
        trace = go.Scatter(
            x=spikesTimes[n].numpy(),
            y=n*np.ones(len(spikesTimes[n])),
            mode="markers",
            marker=dict(size=3, color="black"),
            showlegend=False,
            # hoverinfo="skip",
        )
        fig.add_trace(trace)
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    fig.update_layout(title=title)
    fig.update_layout(
        {
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        }
    )
    return fig

# embedding
def getPlotTrueAndEstimatedEmbeddingParams(trueC, trueD,
                                           estimatedC, estimatedD,
                                           linestyleTrue="solid",
                                           linestyleEstimated="dash",
                                           marker="asterisk",
                                           xlabel="Neuron Index",
                                           ylabel="Coefficient Value"):
    figDic = {
        "data": [],
        "layout": {
            "xaxis": {"title": xlabel},
            "yaxis": {"title": ylabel},
        },
    }
    neuronIndices = np.arange(trueC.shape[0])
    for i in range(estimatedC.shape[1]):
        figDic["data"].append(
            {
                "type": "scatter",
                "name": "true C[{:d}]".format(i),
                "x": neuronIndices,
                "y": trueC[:,i],
                "line": {"dash": linestyleTrue},
                # "marker_symbol": marker,
            },
        )
        figDic["data"].append(
            {
                "type": "scatter",
                "name": "estimated C[{:d}]".format(i),
                "x": neuronIndices,
                "y": estimatedC[:,i],
                "line": {"dash": linestyleEstimated},
                # "marker_symbol": marker,
            },
        )
    figDic["data"].append(
        {
            "type": "scatter",
            "name": "true d",
            "x": neuronIndices,
            "y": trueD[:,0],
            "line": {"dash": linestyleTrue},
            # "marker_symbol": marker,
        },
    )
    figDic["data"].append(
        {
            "type": "scatter",
            "name": "estimated d",
            "x": neuronIndices,
            "y": estimatedD[:,0],
            "line": {"dash": linestyleEstimated},
            # "marker_symbol": marker,
        },
    )
    fig = go.Figure(
        data=figDic["data"],
        layout=figDic["layout"],
    )
    return fig

def getSimulatedEmbeddingPlot(times, samples, means, stds, title, 
                              cbAlpha = 0.2, 
                              cbFillColorPattern="rgba(0,0,255,{:f})", 
                              samplesLineColor="black", 
                              meanLineColor="blue", 
                              xlabel="Time (sec)", 
                              ylabel="Embedding"):
    # tSamples[r], tMeans[r], tSTDs[r],
    # eMean[r], eSTDs[r] \in nNeurons x nSamples
    pio.renderers.default = "browser"
    #
    ci = 1.96*stds
    x = times
    x_rev = x.flip(dims=[0])
    yMeans = means
    ySamples = samples
    yMeans_upper = yMeans + ci
    yMeans_lower = yMeans - ci
    yMeans_lower = yMeans_lower.flip(dims=[0])

    x = x.detach().numpy()
    yMeans = yMeans.detach().numpy()
    ySamples = ySamples.detach().numpy()
    yMeans_upper = yMeans_upper.detach().numpy()
    yMeans_lower = yMeans_lower.detach().numpy()

    traceCB = go.Scatter(
        x=np.concatenate((x, x_rev)),
        y=np.concatenate((yMeans_upper, yMeans_lower)),
        fill="tozerox",
        fillcolor=cbFillColorPattern.format(cbAlpha),
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        name="True",
    )
    traceMean = go.Scatter(
        x=x,
        y=yMeans,
        line=dict(color=meanLineColor),
        mode="lines",
        name="Mean",
        showlegend=True,
    )
    traceSamples = go.Scatter(
        x=x,
        y=ySamples,
        line=dict(color=samplesLineColor),
        mode="lines",
        name="Sample",
        showlegend=True,
    )
    fig = go.Figure()
    fig.add_trace(traceCB)
    fig.add_trace(traceMean)
    fig.add_trace(traceSamples)
    fig.update_yaxes(title_text=ylabel)
    fig.update_xaxes(title_text=xlabel)
    fig.update_layout(title=title)
    return fig

def getPlotTrueAndEstimatedEmbedding(tTimes, tSamples, tMeans, tSTDs,
                                     eTimes, eMeans, eSTDs,
                                     CBalpha = 0.2,
                                     tCBFillColorPattern="rgba(0,0,255,{:f})",
                                     tSamplesLineColor="black",
                                     tMeanLineColor="blue",
                                     eCBFillColorPattern="rgba(255,0,0,{:f})",
                                     eMeanLineColor="red",
                                     xlabel="Time (sec)",
                                     ylabel="Embedding",
                                     title=""):
    # tSamples[r], tMeans[r], tSTDs[r],
    # eMean[r], eSTDs[r] \in nNeurons x nSamples
    pio.renderers.default = "browser"
    #
    eCI = 1.96*eSTDs
    xE = eTimes
    xE_rev = xE.flip(dims=[0])
    yE = eMeans
    yE_upper = yE + eCI
    yE_lower = yE - eCI
    yE_lower = yE_lower.flip(dims=[0])

    xE = xE.detach().numpy()
    yE = yE.detach().numpy()
    yE_upper = yE_upper.detach().numpy()
    yE_lower = yE_lower.detach().numpy()

    tCI = 1.96*tSTDs
    xT = tTimes
    xT_rev = xT.flip(dims=[0])
    yTMeans = tMeans
    yTSamples = tSamples
    yTMeans_upper = yTMeans + tCI
    yTMeans_lower = yTMeans - tCI
    yTMeans_lower = yTMeans_lower.flip(dims=[0])

    xT = xT.detach().numpy()
    yTMeans = yTMeans.detach().numpy()
    yTSamples = yTSamples.detach().numpy()
    yTMeans_upper = yTMeans_upper.detach().numpy()
    yTMeans_lower = yTMeans_lower.detach().numpy()

    traceECB = go.Scatter(
        x=np.concatenate((xE, xE_rev)),
        y=np.concatenate((yE_upper, yE_lower)),
        fill="tozerox",
        fillcolor=eCBFillColorPattern.format(CBalpha),
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        name="Estimated",
    )
    traceEMean = go.Scatter(
        x=xE,
        y=yE,
        # line=dict(color="rgb(0,100,80)"),
        line=dict(color=eMeanLineColor),
        mode="lines",
        name="Estimated Mean",
        showlegend=True,
    )
    traceTCB = go.Scatter(
        x=np.concatenate((xT, xT_rev)),
        y=np.concatenate((yTMeans_upper, yTMeans_lower)),
        fill="tozerox",
        fillcolor=tCBFillColorPattern.format(CBalpha),
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        name="True",
    )
    traceTMean = go.Scatter(
        x=xT,
        y=yTMeans,
        line=dict(color=tMeanLineColor),
        mode="lines",
        name="True Mean",
        showlegend=True,
    )
    traceTSamples = go.Scatter(
        x=xT,
        y=yTSamples,
        line=dict(color=tSamplesLineColor),
        mode="lines",
        name="True Sample",
        showlegend=True,
    )
    fig = go.Figure()
    fig.add_trace(traceECB)
    fig.add_trace(traceEMean)
    fig.add_trace(traceTCB)
    fig.add_trace(traceTMean)
    fig.add_trace(traceTSamples)
    fig.update_yaxes(title_text=ylabel)
    fig.update_xaxes(title_text=xlabel)
    fig.update_layout(title=title)
    return fig

# inducing points
def getPlotTrueAndEstimatedIndPointsLocs(trueIndPointsLocs,
                                         estimatedIndPointsLocs,
                                         linetypeTrue="solid",
                                         linetypeEstimated="dash",
                                         labelTrue="True",
                                         labelEstimated="Estimated",
                                         marker="asterisk",
                                         xlabel="Inducing Point Index",
                                         ylabel="Inducing Point Location"):
    def getTracesOneSetTrueAndEstimatedIndPointsLocs(
        trueIndPointsLocs,
        estimatedIndPointsLocs,
        labelTrue, labelEstimated,
        useLegend):
        traceTrue = go.Scatter(
            y=trueIndPointsLocs,
            mode="lines+markers",
            name=labelTrue,
            line=dict(dash=linetypeTrue),
            showlegend=useLegend)
        traceEstimated = go.Scatter(
            y=estimatedIndPointsLocs,
            mode="lines+markers",
            name=labelEstimated,
            line=dict(dash=linetypeEstimated),
            showlegend=useLegend)
        return traceTrue, traceEstimated

    nLatents = len(trueIndPointsLocs)
    nTrials = trueIndPointsLocs[0].shape[0]
    fig = plotly.subplots.make_subplots(rows=nTrials, cols=nLatents)
    for r in range(nTrials):
        for k in range(nLatents):
            if r==0 and k==nLatents-1:
                useLegend = True
            else:
                useLegend = False
            traceTrue, traceEstimated = getTracesOneSetTrueAndEstimatedIndPointsLocs(trueIndPointsLocs=trueIndPointsLocs[k][r,:,0], estimatedIndPointsLocs=estimatedIndPointsLocs[k][r,:,0], labelTrue=labelTrue, labelEstimated=labelEstimated, useLegend=useLegend)
            fig.add_trace(traceTrue, row=r+1, col=k+1)
            fig.add_trace(traceEstimated, row=r+1, col=k+1)
            fig.update_layout(title="Trial {:d}, Latent {:d}".format(r, k))
    fig.update_yaxes(title_text=ylabel, row=nTrials//2+1, col=1)
    fig.update_xaxes(title_text=xlabel, row=nTrials, col=nLatents//2+1)
    return fig

def getPlotTrueAndEstimatedIndPointsLocsOneTrialOneLatent(
    trueIndPointsLocs,
    estimatedIndPointsLocs,
    title,
    linetypeTrue="solid",
    linetypeEstimated="dash",
    labelTrue="True",
    labelEstimated="Estimated",
    marker="asterisk",
    xlabel="Inducing Point Index",
    ylabel="Inducing Point Location"):

    def getTracesOneSetTrueAndEstimatedIndPointsLocs(
        trueIndPointsLocs,
        estimatedIndPointsLocs,
        labelTrue, labelEstimated,
        useLegend):
        traceTrue = go.Scatter(
            y=trueIndPointsLocs,
            mode="lines+markers",
            name=labelTrue,
            line=dict(dash=linetypeTrue),
            showlegend=useLegend)
        traceEstimated = go.Scatter(
            y=estimatedIndPointsLocs,
            mode="lines+markers",
            name=labelEstimated,
            line=dict(dash=linetypeEstimated),
            showlegend=useLegend)
        return traceTrue, traceEstimated

    fig = go.Figure()
    traceTrue, traceEstimated = getTracesOneSetTrueAndEstimatedIndPointsLocs(trueIndPointsLocs=trueIndPointsLocs, estimatedIndPointsLocs=estimatedIndPointsLocs, labelTrue=labelTrue, labelEstimated=labelEstimated, useLegend=True)
    fig.add_trace(traceTrue)
    fig.add_trace(traceEstimated)
    fig.update_layout(title=title)
    fig.update_yaxes(title_text=ylabel)
    fig.update_xaxes(title_text=xlabel)
    return fig

# variational params
def getPlotTrueAndEstimatedIndPointsMeans(trueIndPointsMeans,
                                          estimatedIndPointsMeans,
                                          linetypeTrue="solid",
                                          linetypeEstimated="dash",
                                          labelTrue="True",
                                          labelEstimated="Estimated",
                                          xlabel="Inducing Point Index",
                                          ylabel="Inducing Point Mean"):
    def getTracesOneSetTrueAndEstimatedIndPointsMeans(
        trueIndPointsMean,
        estimatedIndPointsMean,
        labelTrue, labelEstimated,
        useLegend):
        traceTrue = go.Scatter(
            y=trueIndPointsMean,
            mode="lines+markers",
            name=labelTrue,
            line=dict(dash=linetypeTrue),
            showlegend=useLegend)
        traceEstimated = go.Scatter(
            y=estimatedIndPointsMean,
            mode="lines+markers",
            name=labelEstimated,
            line=dict(dash=linetypeEstimated),
            showlegend=useLegend)
        return traceTrue, traceEstimated

    # trueIndPointsMeans[r][k] \in nInd[k]
    # qMu[k] \in nTrials x nInd[k] x 1
    nTrials = len(trueIndPointsMeans)
    nLatents = len(trueIndPointsMeans[0])
    fig = plotly.subplots.make_subplots(rows=nTrials, cols=nLatents)
    for r in range(nTrials):
        for k in range(nLatents):
            trueIndPointsMean = trueIndPointsMeans[r][k][:,0]
            estimatedIndPointsMean = estimatedIndPointsMeans[k][r,:,0]
            if r==0 and k==nLatents-1:
                useLegend = True
            else:
                useLegend = False
            traceTrue, traceEstimated = getTracesOneSetTrueAndEstimatedIndPointsMeans(trueIndPointsMean=trueIndPointsMean, estimatedIndPointsMean=estimatedIndPointsMean, labelTrue=labelTrue, labelEstimated=labelEstimated, useLegend=useLegend)
            fig.add_trace(traceTrue, row=r+1, col=k+1)
            fig.add_trace(traceEstimated, row=r+1, col=k+1)
            fig.update_layout(title="Trial {:d}, Latent {:d}".format(r, k))
    fig.update_yaxes(title_text=ylabel, row=nTrials//2+1, col=1)
    fig.update_xaxes(title_text=xlabel, row=nTrials, col=nLatents//2+1)
    return fig

def getPlotTrueAndEstimatedIndPointsMeansOneTrialOneLatent(
    trueIndPointsMeans,
    estimatedIndPointsMeans,
    trueIndPointsSTDs,
    estimatedIndPointsSTDs,
    title,
    cbAlpha = 0.2,
    trueCBFillColorPattern="rgba(0,0,255,{:f})",
    trueMeanLineColor="blue",
    estimatedCBFillColorPattern="rgba(255,0,0,{:f})",
    estimatedMeanLineColor="red",
    xlabel="Inducing Point Index",
    ylabel="Inducing Point Mean"):

    indPointsIndices = torch.arange(len(trueIndPointsMeans))

    eCI = 1.96*estimatedIndPointsSTDs
    xE = indPointsIndices
    xE_rev = xE.flip(dims=[0])
    yE = estimatedIndPointsMeans
    yE_upper = yE + eCI
    yE_lower = yE - eCI
    yE_lower = yE_lower.flip(dims=[0])

    xE = xE.detach().numpy()
    yE = yE.detach().numpy()
    yE_upper = yE_upper.detach().numpy()
    yE_lower = yE_lower.detach().numpy()

    tCI = 1.96*trueIndPointsSTDs
    xT = indPointsIndices
    xT_rev = xT.flip(dims=[0])
    yT = trueIndPointsMeans
    yT_upper = yT + tCI
    yT_lower = yT - tCI
    yT_lower = yT_lower.flip(dims=[0])

    xT = xT.detach().numpy()
    yT = yT.detach().numpy()
    yT_upper = yT_upper.detach().numpy()
    yT_lower = yT_lower.detach().numpy()

    traceECB = go.Scatter(
        x=np.concatenate((xE, xE_rev)),
        y=np.concatenate((yE_upper, yE_lower)),
        fill="tozerox",
        fillcolor=estimatedCBFillColorPattern.format(cbAlpha),
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        name="Estimated",
    )
    traceEMean = go.Scatter(
        x=xE,
        y=yE,
        # line=dict(color="rgb(0,100,80)"),
        line=dict(color=estimatedMeanLineColor),
        mode="lines+markers",
        name="Estimated Mean",
        showlegend=True,
    )
    traceTCB = go.Scatter(
        x=np.concatenate((xT, xT_rev)),
        y=np.concatenate((yT_upper, yT_lower)),
        fill="tozerox",
        fillcolor=trueCBFillColorPattern.format(cbAlpha),
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        name="True",
    )
    traceTMean = go.Scatter(
        x=xT,
        y=yT,
        line=dict(color=trueMeanLineColor),
        mode="lines+markers",
        name="True Mean",
        showlegend=True,
    )
    fig = go.Figure()
    fig.add_trace(traceECB)
    fig.add_trace(traceEMean)
    fig.add_trace(traceTCB)
    fig.add_trace(traceTMean)
    fig.update_yaxes(title_text=ylabel)
    fig.update_xaxes(title_text=xlabel)
    fig.update_layout(title=title)
    return fig

#     def getTracesOneSetTrueAndEstimatedIndPointsMeans(
#         trueIndPointsMeans,
#         estimatedIndPointsMeans,
#         labelTrue, labelEstimated,
#         useLegend):
#         traceTrue = go.Scatter(
#             y=trueIndPointsMeans,
#             mode="lines+markers",
#             name=labelTrue,
#             line=dict(dash=linetypeTrue),
#             showlegend=useLegend)
#         traceEstimated = go.Scatter(
#             y=estimatedIndPointsMeans,
#             mode="lines+markers",
#             name=labelEstimated,
#             line=dict(dash=linetypeEstimated),
#             showlegend=useLegend)
#         return traceTrue, traceEstimated
# 
#     # qMu[k] \in nTrials x nInd[k] x 1
#     fig = go.Figure()
#     traceTrue, traceEstimated = getTracesOneSetTrueAndEstimatedIndPointsMeans(trueIndPointsMeans=trueIndPointsMeans, estimatedIndPointsMeans=estimatedIndPointsMeans, labelTrue=labelTrue, labelEstimated=labelEstimated, useLegend=True)
#     fig.add_trace(traceTrue)
#     fig.add_trace(traceEstimated)
#     fig.update_layout(title=title)
#     fig.update_yaxes(title_text=ylabel)
#     fig.update_xaxes(title_text=xlabel)
#     return fig

def getPlotTrueAndEstimatedIndPointsCovs(trueIndPointsCovs,
                                         estimatedIndPointsCovs,
                                         linetypeTrue="solid",
                                         linetypeEstimated="dash",
                                         labelTruePattern="True[:,{:d}]",
                                         labelEstimatedPattern="Estimated[:,{:d}]",
                                         colorsList=plotly.colors.qualitative.Plotly,
                                         xlabel="Inducing Point Index",
                                         ylabel="Inducing Points Covariance"):
    def getTracesOneSetTrueAndEstimatedIndPointsCovs(
        trueIndPointsCov,
        estimatedIndPointsCov,
        labelTruePattern, labelEstimatedPattern,
        useLegend):
        nCols = trueIndPointsCov.shape[1]
        tracesTrue = [[] for i in range(nCols)]
        tracesEstimated = [[] for i in range(nCols)]
        for i in range(nCols):
            color = colorsList[i%len(colorsList)]
            tracesTrue[i] = go.Scatter(
                y=trueIndPointsCov[:,i],
                mode="lines+markers",
                name=labelTruePattern.format(i),
                line=dict(dash=linetypeTrue, color=color),
                showlegend=useLegend)
            tracesEstimated[i] = go.Scatter(
                y=estimatedIndPointsCov[:,i],
                mode="lines+markers",
                name=labelEstimatedPattern.format(i),
                line=dict(dash=linetypeEstimated, color=color),
                showlegend=useLegend)
        return tracesTrue, tracesEstimated

    # trueIndPointsCovs[r][k] \in nInd[k]
    # qMu[k] \in nTrials x nInd[k] x 1
    nTrials = len(trueIndPointsCovs)
    nLatents = len(trueIndPointsCovs[0])
    fig = plotly.subplots.make_subplots(rows=nTrials, cols=nLatents)
    for r in range(nTrials):
        for k in range(nLatents):
            trueIndPointsCov = trueIndPointsCovs[r][k]
            estimatedIndPointsCov = estimatedIndPointsCovs[r][k]
            if r==0 and k==nLatents-1:
                useLegend = True
            else:
                useLegend = False
            tracesTrue, tracesEstimated = getTracesOneSetTrueAndEstimatedIndPointsCovs(trueIndPointsCov=trueIndPointsCov, estimatedIndPointsCov=estimatedIndPointsCov, labelTruePattern=labelTruePattern, labelEstimatedPattern=labelEstimatedPattern, useLegend=useLegend)
            for i in range(len(tracesTrue)):
                fig.add_trace(tracesTrue[i], row=r+1, col=k+1)
                fig.add_trace(tracesEstimated[i], row=r+1, col=k+1)
            fig.update_layout(title="Trial {:d}, Latent {:d}".format(r, k))
    fig.update_yaxes(title_text=ylabel, row=nTrials//2+1, col=1)
    fig.update_xaxes(title_text=xlabel, row=nTrials, col=nLatents//2+1)
    return fig

def getPlotTrueAndEstimatedIndPointsCovsOneTrialOneLatent(
    trueIndPointsCov,
    estimatedIndPointsCov,
    title,
    linetypeTrue="solid",
    linetypeEstimated="dash",
    labelTruePattern="True[:,{:d}]",
    labelEstimatedPattern="Estimated[:,{:d}]",
    colorsList=plotly.colors.qualitative.Plotly,
    xlabel="Inducing Point Index",
    ylabel="Inducing Points Covariance"):

    def getTracesOneSetTrueAndEstimatedIndPointsCovs(
        trueIndPointsCov,
        estimatedIndPointsCov,
        labelTruePattern, labelEstimatedPattern,
        useLegend):

        nColsT = trueIndPointsCov.shape[1]
        tracesTrue = [[] for i in range(nColsT)]
        for i in range(nColsT):
            color = colorsList[i%len(colorsList)]
            tracesTrue[i] = go.Scatter(
                y=trueIndPointsCov[:,i],
                mode="lines+markers",
                name=labelTruePattern.format(i),
                line=dict(dash=linetypeTrue, color=color),
                showlegend=useLegend)

        nColsE = estimatedIndPointsCov.shape[1]
        tracesEstimated = [[] for i in range(nColsE)]
        for i in range(nColsE):
            color = colorsList[i%len(colorsList)]
            tracesEstimated[i] = go.Scatter(
                y=estimatedIndPointsCov[:,i],
                mode="lines+markers",
                name=labelEstimatedPattern.format(i),
                line=dict(dash=linetypeEstimated, color=color),
                showlegend=useLegend)

        return tracesTrue, tracesEstimated

    # trueIndPointsCovs[r][k] \in nInd[k]
    # qMu[k] \in nTrials x nInd[k] x 1
    fig = go.Figure()
    tracesTrue, tracesEstimated = getTracesOneSetTrueAndEstimatedIndPointsCovs(trueIndPointsCov=trueIndPointsCov, estimatedIndPointsCov=estimatedIndPointsCov, labelTruePattern=labelTruePattern, labelEstimatedPattern=labelEstimatedPattern, useLegend=True)
    for i in range(len(tracesTrue)):
        fig.add_trace(tracesTrue[i])
    for i in range(len(tracesEstimated)):
        fig.add_trace(tracesEstimated[i])
    fig.update_layout(title=title)
    fig.update_yaxes(title_text=ylabel)
    fig.update_xaxes(title_text=xlabel)
    return fig

# latents
def getPlotTruePythonAndMatlabLatents(tTimes, tLatents,
                                      pTimes, pMuK, pVarK,
                                      mTimes, mMuK, mVarK,
                                      trialToPlot=0,
                                      xlabel="Time (sec)",
                                      ylabelPattern="Latent {:d}"):
    pio.renderers.default = "browser"
    nLatents = mMuK.shape[2]
    fig = plotly.subplots.make_subplots(rows=nLatents, cols=1, shared_xaxes=True)
    # titles = ["Trial {:d}".format(trialToPlot)] + ["" for i in range(nLatents)]
    title = "Trial {:d}".format(trialToPlot)
    for k in range(nLatents):
        trueToPlot = tLatents[trialToPlot,:,k]

        pMeanToPlot = pMuK[trialToPlot,:,k]
        positiveMSE = torch.mean((trueToPlot-pMeanToPlot)**2)
        negativeMSE = torch.mean((trueToPlot+pMeanToPlot)**2)
        if negativeMSE<positiveMSE:
            pMeanToPlot = -pMeanToPlot
        pCIToPlot = 1.96*(pVarK[trialToPlot,:,k].sqrt())

        mMeanToPlot = mMuK[trialToPlot,:,k]
        positiveMSE = torch.mean((trueToPlot-mMeanToPlot)**2)
        negativeMSE = torch.mean((trueToPlot+mMeanToPlot)**2)
        if negativeMSE<positiveMSE:
            mMeanToPlot = -mMeanToPlot
        mCIToPlot = 1.96*(mVarK[trialToPlot,:,k].sqrt())

        tLatentToPlot = tLatents[trialToPlot,:,k]

        x1 = pTimes
        x1_rev = x1.flip(dims=[0])
        y1 = pMeanToPlot
        y1_upper = y1 + pCIToPlot
        y1_lower = y1 - pCIToPlot
        # y1_lower = y1_lower[::-1] # negative stride not supported in pytorch
        y1_lower = y1_lower.flip(dims=[0])

        x2 = mTimes
        x2_rev = x2.flip(dims=[0])
        y2 = mMeanToPlot
        y2_upper = y2 + mCIToPlot
        y2_lower = y2 - mCIToPlot
        # y2_lower = y2_lower[::-1] # negative stride not supported in pytorch
        y2_lower = y2_lower.flip(dims=[0])

        x3 = tTimes
        y3 = tLatentToPlot

        trace1 = go.Scatter(
            x=np.concatenate((x1, x1_rev)),
            y=np.concatenate((y1_upper, y1_lower)),
            fill="tozerox",
            fillcolor="rgba(255,0,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            name="Python",
        )
        trace2 = go.Scatter(
            x=np.concatenate((x2, x2_rev)),
            y=np.concatenate((y2_upper, y2_lower)),
            fill="tozerox",
            fillcolor="rgba(0,0,255,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Matlab",
            showlegend=False,
        )
        trace3 = go.Scatter(
            x=x1,
            y=y1,
            # line=dict(color="rgb(0,100,80)"),
            line=dict(color="red"),
            mode="lines",
            name="Python",
            showlegend=(k==0),
        )
        trace4 = go.Scatter(
            x=x2,
            y=y2,
            # line=dict(color="rgb(0,176,246)"),
            line=dict(color="blue"),
            mode="lines",
            name="Matlab",
            showlegend=(k==0),
        )
        trace5 = go.Scatter(
            x=x3,
            y=y3,
            line=dict(color="black"),
            mode="lines",
            name="True",
            showlegend=(k==0),
        )
        fig.add_trace(trace1, row=k+1, col=1)
        fig.add_trace(trace2, row=k+1, col=1)
        fig.add_trace(trace3, row=k+1, col=1)
        fig.add_trace(trace4, row=k+1, col=1)
        fig.add_trace(trace5, row=k+1, col=1)
        fig.update_yaxes(title_text=ylabelPattern.format(k+1), row=k+1, col=1)
        # pdb.set_trace()

    fig.update_layout(title_text=title)
    fig.update_xaxes(title_text=xlabel, row=3, col=1)
    return fig

def getPlotTrueAndEstimatedLatents(tTimes, tLatentsSamples, tLatentsMeans, tLatentsSTDs, tIndPointsLocs,
                                   eTimes, eLatentsMeans, eLatentsSTDs, eIndPointsLocs,
                                   trialToPlot=0,
                                   CBalpha = 0.2,
                                   tCBFillColorPattern="rgba(0,0,255,{:f})",
                                   tSamplesLineColor="black",
                                   tMeanLineColor="blue",
                                   eCBFillColorPattern="rgba(255,0,0,{:f})",
                                   eMeanLineColor="red",
                                   tIndPointsLocsColor="rgba(0,0,255,0.5)",
                                   eIndPointsLocsColor="rgba(255,0,0,0.5)",
                                   xlabel="Time (sec)",
                                   ylabelPattern="Trial {:d}"):
    pio.renderers.default = "browser"
    nLatents = eLatentsMeans.shape[2]
    fig = plotly.subplots.make_subplots(rows=nLatents, cols=1, shared_xaxes=True)
    title = ylabelPattern.format(trialToPlot)
    nTrials = len(tLatentsSTDs)
    #
    # latentsMaxs = [1.96*torch.max(tLatentsSTDs[r]).item() for r in range(nTrials)]
    # latentsMaxs.append((torch.max(eLatentsMeans)+1.96*torch.max(eLatentsSTDs)).item())
    # ymax = max(latentsMaxs)
    #
    # latentsMins = [1.96*torch.max(tLatentsSTDs[r]).item() for r in range(nTrials)]
    # latentsMins.append((torch.min(eLatentsMeans)-1.96*torch.max(eLatentsSTDs)).item())
    # ymin = min(latentsMins)
    #
    for k in range(nLatents):
        tSamplesToPlot = tLatentsSamples[trialToPlot][k,:]
        tMeanToPlot = tLatentsMeans[trialToPlot][k,:]
        tSTDToPlot = tLatentsSTDs[trialToPlot][k,:]
        tCIToPlot = 1.96*tSTDToPlot

        eMeanToPlot = eLatentsMeans[trialToPlot,:,k]
        eSTDToPlot = eLatentsSTDs[trialToPlot,:,k]
        positiveMSE = torch.mean((tMeanToPlot-eMeanToPlot)**2)
        negativeMSE = torch.mean((tMeanToPlot+eMeanToPlot)**2)
        if negativeMSE<positiveMSE:
            eMeanToPlot = -eMeanToPlot
        eCIToPlot = 1.96*eSTDToPlot

        ymax = max(torch.max(tMeanToPlot+tCIToPlot), torch.max(eMeanToPlot+eCIToPlot))
        ymin = min(torch.min(tMeanToPlot-tCIToPlot), torch.min(eMeanToPlot-eCIToPlot))

        xE = eTimes
        xE_rev = xE.flip(dims=[0])
        yE = eMeanToPlot
        yE_upper = yE + eCIToPlot
        yE_lower = yE - eCIToPlot
        yE_lower = yE_lower.flip(dims=[0])

        xE = xE.detach().numpy()
        yE = yE.detach().numpy()
        yE_upper = yE_upper.detach().numpy()
        yE_lower = yE_lower.detach().numpy()

        xT = tTimes
        xT_rev = xT.flip(dims=[0])
        yT = tMeanToPlot
        yTSamples = tSamplesToPlot
        yT_upper = yT + tCIToPlot
        yT_lower = yT - tCIToPlot
        yT_lower = yT_lower.flip(dims=[0])

        xT = xT.detach().numpy()
        yT = yT.detach().numpy()
        yTSamples = yTSamples.detach().numpy()
        yT_upper = yT_upper.detach().numpy()
        yT_lower = yT_lower.detach().numpy()

        traceECB = go.Scatter(
            x=np.concatenate((xE, xE_rev)),
            y=np.concatenate((yE_upper, yE_lower)),
            fill="tozerox",
            fillcolor=eCBFillColorPattern.format(CBalpha),
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            name="Estimated",
        )
        traceEMean = go.Scatter(
            x=xE,
            y=yE,
            # line=dict(color="rgb(0,100,80)"),
            line=dict(color=eMeanLineColor),
            mode="lines",
            name="Estimated",
            showlegend=(k==0),
        )
        traceTCB = go.Scatter(
            x=np.concatenate((xT, xT_rev)),
            y=np.concatenate((yT_upper, yT_lower)),
            fill="tozerox",
            fillcolor=tCBFillColorPattern.format(CBalpha),
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=False,
            name="True",
        )
        traceTMean = go.Scatter(
            x=xT,
            y=yT,
            line=dict(color=tMeanLineColor),
            mode="lines",
            name="True",
            showlegend=(k==0),
        )
        traceTSamples = go.Scatter(
            x=xT,
            y=yTSamples,
            line=dict(color=tSamplesLineColor),
            mode="lines",
            name="True",
            showlegend=(k==0),
        )
        fig.add_trace(traceECB, row=k+1, col=1)
        fig.add_trace(traceEMean, row=k+1, col=1)
        fig.add_trace(traceTCB, row=k+1, col=1)
        fig.add_trace(traceTMean, row=k+1, col=1)
        fig.add_trace(traceTSamples, row=k+1, col=1)
        fig.update_yaxes(title_text=ylabelPattern.format(k+1), row=k+1, col=1)

        for n in range(tIndPointsLocs[k].shape[1]):
            fig.add_shape(
                dict(
                    type="line",
                    x0=tIndPointsLocs[k][trialToPlot,n,0],
                    y0=ymin,
                    x1=tIndPointsLocs[k][trialToPlot,n,0],
                    y1=ymax,
                    line=dict(
                        color=tIndPointsLocsColor,
                        width=3
                    ),
                ),
                row=k+1,
                col=1,
            )
            fig.add_shape(
                dict(
                    type="line",
                    x0=eIndPointsLocs[k][trialToPlot,n,0],
                    y0=ymin,
                    x1=eIndPointsLocs[k][trialToPlot,n,0],
                    y1=ymax,
                    line=dict(
                        color=eIndPointsLocsColor,
                        width=3
                    ),
                ),
                row=k+1,
                col=1,
            )
    fig.update_xaxes(title_text=xlabel, row=nLatents, col=1)
    fig.update_layout(title_text=title)
    return fig

def getPlotTrueAndEstimatedLatentsOneTrialOneLatent(
    tTimes, tLatentsSamples, tLatentsMeans, tLatentsSTDs,
    eTimes, eLatentsMeans, eLatentsSTDs,
    title,
    CBalpha = 0.2,
    tCBFillColorPattern="rgba(0,0,255,{:f})",
    tSamplesLineColor="black",
    tMeanLineColor="blue",
    eCBFillColorPattern="rgba(255,0,0,{:f})",
    eMeanLineColor="red",
    xlabel="Time (sec)",
    ylabel="Latent Value"):

    pio.renderers.default = "browser"
    fig = go.Figure()

    tCI = 1.96*tLatentsSTDs
    eCI = 1.96*eLatentsSTDs

    ymax = max(torch.max(tLatentsMeans+tCI), torch.max(eLatentsMeans+eCI))
    ymin = min(torch.min(tLatentsMeans-tCI), torch.min(eLatentsMeans-eCI))

    xE = eTimes
    xE_rev = xE.flip(dims=[0])
    yE = eLatentsMeans
    yE_upper = yE + eCI
    yE_lower = yE - eCI
    yE_lower = yE_lower.flip(dims=[0])

    xE = xE.detach().numpy()
    yE = yE.detach().numpy()
    yE_upper = yE_upper.detach().numpy()
    yE_lower = yE_lower.detach().numpy()

    xT = tTimes
    xT_rev = xT.flip(dims=[0])
    yT = tLatentsMeans
    yTSamples = tLatentsSamples
    yT_upper = yT + tCI
    yT_lower = yT - tCI
    yT_lower = yT_lower.flip(dims=[0])

    xT = xT.detach().numpy()
    yT = yT.detach().numpy()
    yTSamples = yTSamples.detach().numpy()
    yT_upper = yT_upper.detach().numpy()
    yT_lower = yT_lower.detach().numpy()

    traceECB = go.Scatter(
        x=np.concatenate((xE, xE_rev)),
        y=np.concatenate((yE_upper, yE_lower)),
        fill="tozerox",
        fillcolor=eCBFillColorPattern.format(CBalpha),
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        name="Estimated",
    )
    traceEMean = go.Scatter(
        x=xE,
        y=yE,
        # line=dict(color="rgb(0,100,80)"),
        line=dict(color=eMeanLineColor),
        mode="lines",
        name="Estimated",
        showlegend=True,
    )
    traceTCB = go.Scatter(
        x=np.concatenate((xT, xT_rev)),
        y=np.concatenate((yT_upper, yT_lower)),
        fill="tozerox",
        fillcolor=tCBFillColorPattern.format(CBalpha),
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        name="True",
    )
    traceTMean = go.Scatter(
        x=xT,
        y=yT,
        line=dict(color=tMeanLineColor),
        mode="lines",
        name="True",
        showlegend=True,
    )
    traceTSamples = go.Scatter(
        x=xT,
        y=yTSamples,
        line=dict(color=tSamplesLineColor),
        mode="lines",
        name="True",
        showlegend=True,
    )
    fig.add_trace(traceECB)
    fig.add_trace(traceEMean)
    fig.add_trace(traceTCB)
    fig.add_trace(traceTMean)
    fig.add_trace(traceTSamples)

#     for n in range(tIndPointsLocs[k].shape[1]):
#         fig.add_shape(
#             dict(
#                 type="line",
#                 x0=tIndPointsLocs[k][trialToPlot,n,0],
#                 y0=ymin,
#                 x1=tIndPointsLocs[k][trialToPlot,n,0],
#                 y1=ymax,
#                 line=dict(
#                     color=tIndPointsLocsColor,
#                     width=3
#                 ),
#             ),
#             row=k+1,
#             col=1,
#         )
#         fig.add_shape(
#             dict(
#                 type="line",
#                 x0=eIndPointsLocs[k][trialToPlot,n,0],
#                 y0=ymin,
#                 x1=eIndPointsLocs[k][trialToPlot,n,0],
#                 y1=ymax,
#                 line=dict(
#                     color=eIndPointsLocsColor,
#                     width=3
#                 ),
#             ),
#             row=k+1,
#             col=1,
#         )
    fig.update_layout(title_text=title)
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    return fig

def getPlotTrueAndEstimatedLatentsMeans(trueLatentsMeans, 
                                        estimatedLatentsMeans,
                                        trialsTimes, 
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
    fig = plotly.subplots.make_subplots(rows=nTrials, cols=nLatents)
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
    return fig

def getSimulatedLatentsPlot(trialsTimes, latentsSamples, latentsMeans,
                            latentsSTDs, alpha=0.5, marker="x",
                            xlabel="Time (sec)", ylabel="Amplitude",
                            width=1250, height=850,
                            cbFillColorPattern="rgba(0,100,0,{:f})",
                            meanLineColor="rgb(0,100,00)",
                            samplesLineColor="rgb(0,0,0)"):
    nTrials = len(latentsSamples)
    nLatents = latentsSamples[0].shape[0]
    subplotsTitles = ["trial={:d}, latent={:d}".format(r, k) for r in range(nTrials) for k in range(nLatents)]
    fig = plotly.subplots.make_subplots(rows=nTrials, cols=nLatents, subplot_titles=subplotsTitles)
    for r in range(nTrials):
        t = trialsTimes[r].numpy()
        t_rev = t[::-1]
        for k in range(nLatents):
            samples = latentsSamples[r][k,:].numpy()
            mean = latentsMeans[r][k,:].numpy()
            std = latentsSTDs[r][k,:].numpy()
            upper = mean+1.96*std
            lower = mean-1.96*std
            lower_rev = lower[::-1]

            traceCB = go.Scatter(
                x=np.concatenate((t, t_rev)),
                y=np.concatenate((upper, lower_rev)),
                fill="tozerox",
                fillcolor=cbFillColorPattern.format(alpha),
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
            )
            traceMean = go.Scatter(
                x=t,
                y=mean,
                line=dict(color=meanLineColor),
                mode="lines",
                showlegend=False,
            )
            traceSamples = go.Scatter(
                x=t,
                y=samples,
                line=dict(color=samplesLineColor),
                mode="lines",
                showlegend=False,
            )
            fig.add_trace(traceCB, row=r+1, col=k+1)
            fig.add_trace(traceMean, row=r+1, col=k+1)
            fig.add_trace(traceSamples, row=r+1, col=k+1)
            if r==nTrials-1 and k==math.floor(nLatents/2):
                fig.update_xaxes(title_text=xlabel, row=r+1, col=k+1)
            if r==math.floor(nTrials/2) and k==0:
                fig.update_yaxes(title_text=ylabel, row=r+1, col=k+1)
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
    )
    return fig

def getSimulatedLatentPlot(times, latentSamples, latentMeans,
                            latentSTDs, title, alpha=0.2, marker="x",
                            xlabel="Time (sec)", ylabel="Value",
                            cbFillColorPattern="rgba(0,0,255,{:f})",
                            meanLineColor="rgb(0,0,255)",
                            samplesLineColor="rgb(0,0,0)"):
    t = times.numpy()
    t_rev = t[::-1]
    samples = latentSamples.numpy()
    mean = latentMeans.numpy()
    std = latentSTDs.numpy()
    upper = mean+1.96*std
    lower = mean-1.96*std
    lower_rev = lower[::-1]

    traceCB = go.Scatter(
        x=np.concatenate((t, t_rev)),
        y=np.concatenate((upper, lower_rev)),
        fill="tozerox",
        fillcolor=cbFillColorPattern.format(alpha),
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
    )
    traceMean = go.Scatter(
        x=t,
        y=mean,
        line=dict(color=meanLineColor),
        mode="lines",
        showlegend=True,
        name="Mean",
    )
    traceSamples = go.Scatter(
        x=t,
        y=samples,
        line=dict(color=samplesLineColor),
        mode="lines",
        showlegend=True,
        name="Sample",
    )
    fig = go.Figure()
    fig.add_trace(traceCB)
    fig.add_trace(traceMean)
    fig.add_trace(traceSamples)
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)
    fig.update_layout(title=title)
    return fig

# kernels
def getPlotTrueAndEstimatedKernelsParams(trueKernels, estimatedKernelsParams,
                                         colorTrue="blue",
                                         colorEstimated="red",
                                         trueLegend="True",
                                         estimatedLegend="Estimated"):
    nLatents = len(trueKernels)
    titles = ["Kernel {:d}: {:s}".format(i, trueKernels[i].__class__.__name__) for i in range(nLatents)]
    fig = plotly.subplots.make_subplots(rows=nLatents, cols=1, subplot_titles=titles)
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
    return fig

def getPlotTrueAndEstimatedKernelsParamsOneLatent(
    trueKernel,
    estimatedKernelParams,
    title,
    colorTrue="blue",
    colorEstimated="red",
    trueLegend="True",
    estimatedLegend="Estimated"):

    fig = go.Figure()
    namedParams = trueKernel.getNamedParams()
    del namedParams["scale"]
    labels = list(namedParams.keys())
    trueParams = [z.item() for z in list(namedParams.values())]
    estimatedParams = estimatedKernelParams.tolist()

    traceTrue = go.Bar(x=labels, y=trueParams, name=trueLegend, marker_color=colorTrue, showlegend=True)
    traceEstimated = go.Bar(x=labels, y=estimatedParams, name=estimatedLegend, marker_color=colorEstimated, showlegend=True)
    fig.add_trace(traceTrue)
    fig.add_trace(traceEstimated)
    fig.update_yaxes(title_text="Parameter Value")
    fig.update_layout(title=title)
    return fig

def getPlotTruePythonAndMatlabKernelsParams(kernelsTypes,
                                            trueKernelsParams,
                                            pythonKernelsParams,
                                            matlabKernelsParams,
                                            colorTrue="blue",
                                            colorPython="red",
                                            colorMatlab="green",
                                            trueLegend="True",
                                            pythonLegend="Python",
                                            matlabLegend="Matlab"):
    nLatents = len(trueKernelsParams)
    titles = ["Kernel {:d}: {:s}".format(k, kernelsTypes[k]) for k in range(nLatents)]
    fig = plotly.subplots.make_subplots(rows=nLatents, cols=1, subplot_titles=titles)
    for k in range(nLatents):
        trueParams = trueKernelsParams[k].tolist()
        pythonParams = pythonKernelsParams[k].tolist()
        matlabParams = matlabKernelsParams[k].tolist()
        if k==0:
            showLegend = True
        else:
            showLegend = False

        if kernelsTypes[k]=="PeriodicKernel":
            labels = ["Length Scale", "Period"]
        elif kernelsTypes[k]=="ExponentialQuadraticKernel":
            labels = ["Length Scale"]
        else:
            raise RuntimeError("Invalid kernel type {:s}".format(kernelsTypes[k]))

        traceTrue = go.Bar(x=labels, y=trueParams, name=trueLegend, marker_color=colorTrue, showlegend=showLegend)
        tracePython = go.Bar(x=labels, y=pythonParams, name=pythonLegend, marker_color=colorPython, showlegend=showLegend)
        traceMatlab = go.Bar(x=labels, y=matlabParams, name=matlabLegend, marker_color=colorMatlab, showlegend=showLegend)
        fig.append_trace(traceTrue, k+1, 1)
        fig.append_trace(tracePython, k+1, 1)
        fig.append_trace(traceMatlab, k+1, 1)
    fig.update_yaxes(title_text="Parameter Value", row=nLatents//2+1, col=1)
    return fig

def getPlotLowerBoundVsTwoParamsParam(param1Values,
                                      param2Values,
                                      lowerBoundValues,
                                      refParam1,
                                      refParam2,
                                      refParamsLowerBound,
                                      refParamText,
                                      title,
                                      lowerBoundQuantile = 0.5,
                                      param1Label="Parameter 1",
                                      param2Label="Parameter 2",
                                      lowerBoundLabel="Lower Bound",
                                      markerSize=3.0,
                                      markerOpacity=0.8,
                                      markerColorscale="Viridis",
                                      zMin=None, zMax=None,
                                     ):
    data = {"x": param1Values, "y": param2Values, "z": lowerBoundValues}
    df = pd.DataFrame(data)
    if zMin is None:
        zMin = df.z.quantile(lowerBoundQuantile)
    if zMax is None:
        zMax = df.z.max()
    dfTrimmed = df[df.z>zMin]
    # fig = go.Figure(data=[go.Scatter3d(x=param1Values, y=param2Values, z=lowerBoundValues, mode="markers")])
#     fig = go.Figure(data=[go.Scatter3d(x=dfTrimmed.x, y=dfTrimmed.y, z=dfTrimmed.z, mode="markers")])
    # fig.update_layout(scene=dict(zaxis=dict(range=[df.z.max()-1000,df.z.max()+500],)),width=700,)
    # fig = px.scatter_3d(dfTrimmed, x='x', y='y', z='z')
    fig = go.Figure(data=[go.Scatter3d(x=dfTrimmed.x, y=dfTrimmed.y,
                                       z=dfTrimmed.z, mode="markers",
                                       marker=dict(size=markerSize,
                                                   color=dfTrimmed.z,
                                                   colorscale=markerColorscale,
                                                   opacity=markerOpacity)) ])

#     fig = go.Figure(go.Mesh3d(x=dfTrimmed.x, y=dfTrimmed.y, z=dfTrimmed.z))
#     fig.add_trace(
#         go.Scatter3d(
#             x=[refParam1],
#             y=[refParam2],
#             z=[refParamsLowerBound],
#             type="scatter3d", text=[refParamText], mode="text",
#         )
#     )
    fig.update_layout(title=title, scene = dict(xaxis_title = param1Label, yaxis_title = param2Label, zaxis_title = lowerBoundLabel,))
#     fig.update_layout(scene = dict(zaxis = dict(range=[zMin,zMax]),))
#     fig.update_layout(scene = dict(zaxis=dict(range=[df.z.max()-1000,df.z.max()+500],),),)
#     pio.renderers.default = "browser"
#     fig.show()
#     pdb.set_trace()
    return fig

# CIF
def getPlotTruePythonAndMatlabCIFs(tTimes, tCIF, tLabel,
                                   pTimes, pCIF, pLabel,
                                   mTimes, mCIF, mLabel,
                                   xlabel="Time (sec)",
                                   ylabel="CIF",
                                   title=""
                                   ):
    pio.renderers.default = "browser"
    figDic = {
        "data": [],
        "layout": {
            "xaxis": {"title": xlabel},
            "yaxis": {"title": ylabel},
            "title": {"text": title},
        },
    }
    figDic["data"].append(
            {
            "type": "scatter",
            "name": tLabel,
            "x": tTimes,
            "y": tCIF,
        },
    )
    figDic["data"].append(
            {
            "type": "scatter",
            "name": pLabel,
            "x": pTimes,
            "y": pCIF,
        },
    )
    figDic["data"].append(
            {
            "type": "scatter",
            "name": mLabel,
            "x": mTimes,
            "y": mCIF,
        },
    )
    fig = go.Figure(
        data=figDic["data"],
        layout=figDic["layout"],
    )
    return fig

def getPlotSimulatedAndEstimatedCIFs(tTimes, tCIF, tLabel, eTimes, eCIF, eLabel, xlabel="Time (sec)", ylabel="CIF", title=""):
    pio.renderers.default = "browser"
    figDic = {
        "data": [],
        "layout": {
            "xaxis": {"title": xlabel},
            "yaxis": {"title": ylabel},
            "title": {"text": title},
        },
    }
    figDic["data"].append(
            {
            "type": "scatter",
            "name": tLabel,
            "x": tTimes,
            "y": tCIF,
        },
    )
    figDic["data"].append(
            {
            "type": "scatter",
            "name": eLabel,
            "x": eTimes,
            "y": eCIF,
        },
    )
    fig = go.Figure(
        data=figDic["data"],
        layout=figDic["layout"],
    )
    return fig

def getPlotCIF(times, values, title="", xlabel="Time (sec)", ylabel="Conditional Intensity Function"):
    figDic = {
        "data": [],
        "layout": {
            "title": title,
            "xaxis": {"title": xlabel},
            "yaxis": {"title": ylabel},
        },
    }
    figDic["data"].append(
        {
            "type": "scatter",
            "x": times,
            "y": values,
        },
    )
    fig = go.Figure(
        data=figDic["data"],
        layout=figDic["layout"],
    )
    return fig

# Lower bound
def getPlotLowerBoundHist(lowerBoundHist, elapsedTimeHist=None, xlabelIterNumber="Iteration Number", xlabelElapsedTime="Elapsed Time (sec)", ylabel="Lower Bound", marker="cross", linestyle="solid", figFilename=None):
    if elapsedTimeHist is None:
        trace = go.Scatter(
            y=lowerBoundHist,
            mode="lines+markers",
            line={"color": "red", "dash": linestyle},
            marker={"symbol": marker},
            showlegend=False,
        )
        xlabel = xlabelIterNumber
    else:
        trace = go.Scatter(
            x=elapsedTimeHist,
            y=lowerBoundHist,
            mode="lines+markers",
            line={"color": "red", "dash": linestyle},
            marker_symbol=marker,
            showlegend=False,
        )
        xlabel = xlabelElapsedTime
    fig = go.Figure()
    fig.add_trace(trace)
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text="Lower Bound")
    return fig

def getPlotLowerBoundVsOneParam(paramValues, lowerBoundValues, refParams, title, yMin, yMax, lowerBoundLineColor, refParamsLineColors, percMargin=0.1, xlab="Parameter Value", ylab="Lower Bound"):
    if math.isinf(yMin):
        yMin = lowerBoundValues.min()
    if math.isinf(yMax):
        yMax = lowerBoundValues.max()
    margin = percMargin*max(abs(yMin), abs(yMax))
    yMin = yMin - margin
    yMax = yMax + margin

    layout = {
        "title": title,
        "xaxis": {"title": xlab},
        # "yaxis": {"title": "Lower Bound"},
        "yaxis": {"title": ylab, "range": [yMin, yMax]},
    }
    data = []
    data.append(
        {
            "type": "scatter",
            "mode": "lines+markers",
            # "mode": "markers",
            "x": paramValues,
            "y": lowerBoundValues,
            "marker": dict(color=lowerBoundLineColor),
            "line": dict(color=lowerBoundLineColor),
            "name": "lower bound",
        },
    )
    fig = go.Figure(
        data=data,
        layout=layout,
    )
    for i in range(len(refParams)):
        fig.add_shape(
            # Line Vertical
            dict(
                type="line",
                x0=refParams[i],
                y0=yMin,
                x1=refParams[i],
                y1=yMax,
                line=dict(
                    color=refParamsLineColors[i],
                    width=3
                )
        ))
    return fig
