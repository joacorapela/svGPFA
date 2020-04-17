
import pdb
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.offline

def plotResROCAnalysis(fpr, tpr, auc, figFilename, title="",
                       colorROC="red", colorRef="black",
                       linestyleROC="-", linestyleRef="--",
                       labelPattern="ROC curve (area={:0.2f})",
                       xlabel="False Positive Rate",
                       ylabel="True Positive Rate",
                       legendLoc="lower right"):
    plt.figure()
    plt.plot(fpr, tpr, color=colorROC, linestyle=linestyleROC, label=labelPattern.format(auc))
    plt.plot([0, 1], [0, 1], color=colorRef, linestyle=linestyleRef)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=legendLoc)
    plt.savefig(fname=figFilename)
    plt.show()

def plotACF(acf, confint, Fs, figFilename, title="", xlabel="Lag (sec)", ylabel="ACF", colorACF="black", colorConfint="red", colorRef="gray", linestyleACF="-", linestyleConfint=":", linestyleRef=":"):
    acf[0] = None
    confint[0,:] = None

    time = np.arange(len(acf))/Fs
    plt.figure()
    plt.plot(time, acf, color=colorACF, linestyle=linestyleACF)
    plt.plot(time, confint[:,0], color=colorConfint, linestyle=linestyleConfint)
    plt.plot(time, confint[:,1], color=colorConfint, linestyle=linestyleConfint)
    plt.axhline(y=0, color=colorRef, linestyle=linestyleRef)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fname=figFilename)

def plotScatter1Lag(x, figFilename, title="", xlabel="x[t-1]", ylabel="x[t]"):
    plt.figure()
    plt.scatter(x[:-1], x[1:])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fname=figFilename)

def plotSimulatedAndEstimatedCIFs(times, simCIFValues, estCIFValues,
                                  figFilename,
                                  title="",
                                  labelSimulated="True",
                                  labelEstimated="Estimated",
                                  xlabel="Time (sec)",
                                  ylabel="Conditional Intensity Function"):
    plt.figure()
    plt.plot(times, simCIFValues, label=labelSimulated)
    plt.plot(times, estCIFValues, label=labelEstimated)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(fname=figFilename)

def plotCIF(times, values, figFilename, title="", xlabel="Time (sec)",
            ylabel="Conditional Intensity Function"):
    plt.figure()
    plt.plot(times, values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(fname=figFilename)

def plotResKSTestTimeRescalingAnalyticalCorrection(
    sUTRISIs, uCDF, cb, figFilename,
    title="", dataColor="blue", cbColor="red", dataLinestyle="solid",
    dataMarker="*", cbLinestyle="dashed",
    ylabel="Empirical CDF", xlabel="Model CDF"):

    plt.figure()
    plt.plot(sUTRISIs, uCDF, color=dataColor, linestyle=dataLinestyle, marker=dataMarker)
    plt.plot([0, 1-cb], [cb, 1], color=cbColor, linestyle=cbLinestyle)
    plt.plot([cb, 1], [0, 1-cb], color=cbColor, linestyle=cbLinestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(figFilename)

def plotDifferenceCDFs(
    sUTRISIs, uCDF, cb, figFilename,
    title="", dataColor="blue", cbColor="red", dataLinestyle="solid",
    dataMarker="*", cbLinestyle="dashed",
    ylabel="Difference", xlabel="CDF"):

    plt.figure()
    plt.plot(uCDF, sUTRISIs-uCDF, color=dataColor, linestyle=dataLinestyle, marker=dataMarker)
    plt.plot([0, 1], [cb, cb], color=cbColor, linestyle=cbLinestyle)
    plt.plot([0, 1], [-cb, -cb], color=cbColor, linestyle=cbLinestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(figFilename)

def plotResKSTestTimeRescalingNumericalCorrection(
    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy,
    cb, figFilename, title="",
    dataColor="blue", cbColor="red", refColor="black",
    estECDFcolor="magenta", simECDFcolor="cyan",
    estECDFmarker="+", simECDFmarker="*",
    dataLinestyle="solid", cbLinestyle="dashed", refLinestyle="solid",
    dataMarker="o",
    ylabel="Empirical Cumulative Distribution Function",
    xlabel="Rescaled Time",
    diffLabel="Difference", estECDFlabel="Estimated",
    simECDFlabel="True" ):

    plt.figure()
    plt.plot(diffECDFsX, diffECDFsY, color=dataColor, marker=dataMarker, linestyle=dataLinestyle, label=diffLabel)
    plt.scatter(estECDFx, estECDFy, color=estECDFcolor, marker=estECDFmarker, label=estECDFlabel)
    plt.scatter(simECDFx, simECDFy, color=simECDFcolor, marker=simECDFmarker, label=simECDFlabel)
    plt.axhline(0, color=refColor, linestyle=refLinestyle)
    plt.axhline(cb, color=cbColor, linestyle=cbLinestyle)
    plt.axhline(-cb, color=cbColor, linestyle=cbLinestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(figFilename)

def getSimulatedSpikeTimesPlot(spikesTimes, figFilename, xlabel="Time (sec)", ylabel="Neuron", titlePattern="Trial {:d}"):
    nTrials = len(spikesTimes)
    sqrtNTrials = math.sqrt(nTrials)
    # nrow = math.floor(sqrtNTrials)
    # ncol = math.ceil(sqrtNTrials)
    # f, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True)
    plt.figure()
    f, axs = plt.subplots(nTrials, 1, sharex=True, sharey=True, squeeze=False)
    for r in range(nTrials):
        # row = r//ncol
        # col = r%ncol
        # axs[row, col].eventplot(positions=spikesTimes[r])
        # axs[row, col].set_xlabel(xlabel)
        # axs[row, col].set_ylabel(ylabel)
        # axs[row, col].set_title(titlePattern.format(r))
        row = r
        col = 0
        axs[row, col].eventplot(positions=spikesTimes[r])
        axs[row, col].set_xlabel(xlabel)
        axs[row, col].set_ylabel(ylabel)
        axs[row, col].set_title(titlePattern.format(r))
    plt.savefig(figFilename)
    return f

def getSimulatedLatentsPlot(trialsTimes, latentsSamples, latentsMeans, latentsSTDs, figFilename, alpha=0.5, marker="x", xlabel="Time (sec)", ylabel="Amplitude"):
    nTrials = len(latentsSamples)
    nLatents = latentsSamples[0].shape[0]
    plt.figure()
    f, axs = plt.subplots(nTrials, nLatents, sharex=False, sharey=False, squeeze=False)
    for r in range(nTrials):
        t = trialsTimes[r]
        for k in range(nLatents):
            latentSamples = latentsSamples[r][k,:]
            mean = latentsMeans[r][k,:]
            std = latentsSTDs[r][k,:]
            axs[r,k].plot(t, latentSamples, marker=marker)
            axs[r,k].fill_between(t, mean-1.96*std, mean+1.96*std, alpha=alpha)
            axs[r,k].set_xlabel(xlabel)
            axs[r,k].set_ylabel(ylabel)
            axs[r,k].set_title("r={}, k={}".format(r, k))
    plt.savefig(figFilename)
    return f

def plotEstimatedLatents(fig, times, muK, varK, indPointsLocs, title="", figFilename=None):
    nTrials = muK.shape[0]
    nLatents = muK.shape[2]
    timesToPlot = times.numpy()
    # f, axes = plt.subplots(nTrials, nLatents, sharex=True, squeeze=False)
    # f, axes = plt.subplots(nTrials, nLatents, sharex=True, squeeze=False)
    plt.clf()
    index = 1
    for r in range(nTrials):
        for k in range(nLatents):
            muKToPlot = muK[r,:,k]
            hatCIToPlot = varK[r,:,k]
            # axes[r,k].plot(timesToPlot, muKToPlot, color="blue")
            # axes[r,k].fill_between(timesToPlot, muKToPlot-hatCIToPlot, muKToPlot+hatCIToPlot, color="lightblue")
            ax = fig.add_subplot(nTrials, nLatents, index)
            ax.plot(timesToPlot, muKToPlot, color="blue")
            if(r==0 and k==0):
                plt.title(title)
            ax.fill_between(timesToPlot, muKToPlot-hatCIToPlot, muKToPlot+hatCIToPlot, color="lightblue")
            for i in range(indPointsLocs[k].shape[1]):
                # axes[r,k].axvline(x=indPointsLocs[k][r,i, 0], color="red")
                ax.axvline(x=indPointsLocs[k][r,i, 0], color="red")
            index += 1
    # axes[r//2,0].set_ylabel("Latent")
    # axes[-1,k//2].set_xlabel("Time (sec)")
    # axes[0,-1].legend()
    ax = fig.add_subplot(nTrials, nLatents, 1)
    ax.set_ylabel("Latent")
    ax.set_xlabel("Time (sec)")
    ax.legend()
    # plt.xlim(left=torch.min(timesToPlot)-1, right=torch.max(timesToPlot)+1)
    if figFilename is not None:
        plt.savefig(fname=figFilename)

def plotLowerBoundHist(lowerBoundHist, elapsedTimeHist=None, xlabelIterNumber="Iteration Number", xlabelElapsedTime="Elapsed Time (sec)", ylabel="Lower Bound", marker="x", linestyle="-", figFilename=None):
    plt.figure()
    if elapsedTimeHist is None:
        plt.plot(lowerBoundHist, marker=marker, linestyle=linestyle)
        plt.xlabel(xlabelIterNumber)
    else:
        plt.plot(elapsedTimeHist, lowerBoundHist, marker=marker, linestyle=linestyle)
        plt.xlabel(xlabelElapsedTime)
    plt.ylabel(ylabel)
    if figFilename is not None:
        plt.savefig(fname=figFilename)
    plt.show()

def plotTrueAndEstimatedLatents(timesEstimatedValues, muK, varK, indPointsLocs, timesTrueValues, trueLatents, trueLatentsMeans, trueLatentsSTDs, trialToPlot=0, figFilename=None):
    nLatents = muK.shape[2]
    plt.figure()
    f, axes = plt.subplots(nLatents, 1, sharex=True, squeeze=False)
    title = "Trial {:d}".format(trialToPlot)
    axes[0,0].set_title(title)
    for k in range(nLatents):
        trueLatentsToPlot = trueLatents[trialToPlot][k].detach()
        trueMeanToPlot = trueLatentsMeans[trialToPlot][k].detach()
        trueCIToPlot = 1.96*(trueLatentsSTDs[trialToPlot][k]).detach()
        hatMeanToPlot = muK[trialToPlot,:,k].detach()
        # positiveMSE = torch.mean((trueMeanToPlot-hatMeanToPlot)**2).detach()
        # negativeMSE = torch.mean((trueMeanToPlot+hatMeanToPlot)**2).detach()
        # if negativeMSE<positiveMSE:
        #     hatMeanToPlot = -hatMeanToPlot
        hatCIToPlot = 1.96*(varK[trialToPlot,:,k].sqrt()).detach()
        axes[k,0].plot(timesTrueValues, trueLatentsToPlot, label="true sampled", color="black")
        axes[k,0].plot(timesTrueValues, trueMeanToPlot, label="true mean", color="gray")
        axes[k,0].fill_between(timesTrueValues, trueMeanToPlot-trueCIToPlot, trueMeanToPlot+trueCIToPlot, color="lightgray")
        axes[k,0].plot(timesEstimatedValues, hatMeanToPlot, label="estimated", color="blue")
        axes[k,0].fill_between(timesEstimatedValues, hatMeanToPlot-hatCIToPlot, hatMeanToPlot+hatCIToPlot, color="lightblue")
        for i in range(indPointsLocs[k].shape[1]):
            axes[k,0].axvline(x=indPointsLocs[k][trialToPlot,i, 0], color="red")
        axes[k,0].set_ylabel("Latent %d"%(k))
    axes[-1,0].set_xlabel("Time (sec)")
    axes[-1,0].legend()
    allIndPointsLocs = torch.cat([indPointsLocs[k][trialToPlot,:,0] for k in range(len(indPointsLocs))])
    # allTimes = torch.cat((timesTrueValues, timesEstimatedValues, allIndPointsLocs))
    # plt.xlim(left=torch.min(allTimes), right=torch.max(allTimes))
    if figFilename is not None:
        plt.savefig(fname=figFilename)
    plt.show()

def plotTruePythonAndMatlabLatents(tTimes, tLatents,
                                   pTimes, pMuK, pVarK,
                                   mTimes, mMuK, mVarK,
                                   trialToPlot=0, figFilenamePattern=None):
    figFilename = figFilenamePattern.format(trialToPlot)
    nLatents = mMuK.shape[2]
    plt.figure()
    f, axes = plt.subplots(nLatents, 1, sharex=True)
    title = "Trial {:d}".format(trialToPlot)
    axes[0].set_title(title)
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

        axes[k].plot(tTimes, trueToPlot, label="True", color="black")
        axes[k].plot(pTimes, pMeanToPlot, label="Python", color="darkblue")
        axes[k].plot(mTimes, mMeanToPlot, label="Matlab", color="darkorange")
        axes[k].fill_between(pTimes, (pMeanToPlot-pCIToPlot), (pMeanToPlot+pCIToPlot), color="blue")
        axes[k].fill_between(mTimes, mMeanToPlot-mCIToPlot, mMeanToPlot+mCIToPlot, color="orange")
        axes[k].set_ylabel("Latent %d"%(k))
    axes[-1].set_xlabel("Sample")
    axes[-1].legend()
    plt.xlim(left=torch.min(tTimes)-1, right=torch.max(tTimes)+1)
    if figFilename is not None:
        plt.savefig(fname=figFilename)
    plt.show()

def plotTruePythonAndMatlabLatentsPlotly(tTimes, tLatents,
                                         pTimes, pMuK, pVarK,
                                         mTimes, mMuK, mVarK,
                                         trialToPlot=0,
                                         staticFigFilenamePattern=None,
                                         dynamicFigFilenamePattern=None,
                                         xlabel="Time (sec)",
                                         ylabelPattern="Latent {:d}"):
    pio.renderers.default = "browser"
    staticFigFilename = staticFigFilenamePattern.format(trialToPlot)
    dynamicFigFilename = dynamicFigFilenamePattern.format(trialToPlot)
    nLatents = mMuK.shape[2]
    fig = make_subplots(rows=nLatents, cols=1, shared_xaxes=True)
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
            fill='tozerox',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            name='Python',
        )
        trace2 = go.Scatter(
            x=np.concatenate((x2, x2_rev)),
            y=np.concatenate((y2_upper, y2_lower)),
            fill='tozerox',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Matlab',
            showlegend=False,
        )
        trace3 = go.Scatter(
            x=x1,
            y=y1,
            # line=dict(color='rgb(0,100,80)'),
            line=dict(color='red'),
            mode='lines',
            name='Python',
            showlegend=(k==0),
        )
        trace4 = go.Scatter(
            x=x2,
            y=y2,
            # line=dict(color='rgb(0,176,246)'),
            line=dict(color='blue'),
            mode='lines',
            name='Matlab',
            showlegend=(k==0),
        )
        trace5 = go.Scatter(
            x=x3,
            y=y3,
            line=dict(color='black'),
            mode='lines',
            name='True',
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
    fig.write_image(staticFigFilename)
    plotly.offline.plot(fig, filename=dynamicFigFilename)
    # fig.show()
    # pdb.set_trace()

