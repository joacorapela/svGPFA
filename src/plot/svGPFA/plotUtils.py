
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

def getSimulatedSpikeTimesPlot(spikesTimes, figFilename, xlabel="Time (sec)", ylabel="Neuron", titlePattern="Trial {:d}"):
    nTrials = len(spikesTimes)
    sqrtNTrials = math.sqrt(nTrials)
    # nrow = math.floor(sqrtNTrials)
    # ncol = math.ceil(sqrtNTrials)
    # f, axs = plt.subplots(nrow, ncol, sharex=True, sharey=True)
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
    f, axs = plt.subplots(nTrials, nLatents, sharex=True, sharey=True, squeeze=False)
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

def plotEstimatedLatents(times, muK, varK, indPointsLocs, trialToPlot=0, figFilename=None):
    nLatents = muK.shape[2]
    timesToPlot = times.numpy()
    f, axes = plt.subplots(nLatents, 1, sharex=True)
    for k in range(nLatents):
        muKToPlot = muK[trialToPlot,:,k].detach().numpy()
        hatCIToPlot = varK[trialToPlot,:,k].sqrt().detach().numpy()
        axes[k].plot(timesToPlot, muKToPlot, label="estimated", color="blue")
        axes[k].fill_between(timesToPlot, muKToPlot-hatCIToPlot, 
                             muKToPlot+hatCIToPlot, color="lightblue")
        for i in range(indPointsLocs[k].shape[1]):
            axes[k].axvline(x=indPointsLocs[k][trialToPlot,i, 0], color="red")
        axes[k].set_ylabel("Latent %d"%(k))
    axes[-1].set_xlabel("Sample")
    axes[-1].legend()
    plt.xlim(left=torch.min(timesToPlot)-1, right=torch.max(timesToPlot)+1)
    if figFilename is not None:
        plt.savefig(fname=figFilename)
    plt.show()

def plotLowerBoundHist(lowerBoundHist, elapsedTimeHist=None, xlabelIterNumber="Iteration Number", xlabelElapsedTime="Elapsed Time (sec)", ylabel="Lower Bound", marker="x", linestyle="-", figFilename=None):
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

def plotTrueAndEstimatedLatents(times, muK, varK, indPointsLocs, trueLatents, trueLatentsMeans, trueLatentsSTDs, trialToPlot=0, figFilename=None):
    nLatents = muK.shape[2]
    timesToPlot = times
    f, axes = plt.subplots(nLatents, 1, sharex=True, squeeze=False)
    title = "Trial {:d}".format(trialToPlot)
    axes[0,0].set_title(title)
    for k in range(nLatents):
        trueLatentsToPlot = trueLatents[trialToPlot][k].detach()
        trueMeanToPlot = trueLatentsMeans[trialToPlot][k].detach()
        trueCIToPlot = 1.96*(trueLatentsSTDs[trialToPlot][k]).detach()
        hatMeanToPlot = muK[trialToPlot,:,k].detach()
        positiveMSE = torch.mean((trueMeanToPlot-hatMeanToPlot)**2).detach()
        negativeMSE = torch.mean((trueMeanToPlot+hatMeanToPlot)**2).detach()
        if negativeMSE<positiveMSE:
            hatMeanToPlot = -hatMeanToPlot
        hatCIToPlot = 1.96*(varK[trialToPlot,:,k].sqrt()).detach()
        axes[k,0].plot(timesToPlot, trueLatentsToPlot, label="true sampled", color="black")
        axes[k,0].plot(timesToPlot, trueMeanToPlot, label="true mean", color="gray")
        axes[k,0].fill_between(timesToPlot, trueMeanToPlot-trueCIToPlot, trueMeanToPlot+trueCIToPlot, color="lightgray")
        axes[k,0].plot(timesToPlot, hatMeanToPlot, label="estimated", color="blue")
        axes[k,0].fill_between(timesToPlot, hatMeanToPlot-hatCIToPlot, hatMeanToPlot+hatCIToPlot, color="lightblue")
        for i in range(indPointsLocs[k].shape[1]):
            axes[k,0].axvline(x=indPointsLocs[k][trialToPlot,i, 0], color="red")
        axes[k,0].set_ylabel("Latent %d"%(k))
    axes[-1,0].set_xlabel("Sample")
    axes[-1,0].legend()
    plt.xlim(left=torch.min(timesToPlot)-1, right=torch.max(timesToPlot)+1)
    if figFilename is not None:
        plt.savefig(fname=figFilename)
    plt.show()

def plotTruePythonAndMatlabLatents(tTimes, tLatents,
                                   pTimes, pMuK, pVarK,
                                   mTimes, mMuK, mVarK,
                                   trialToPlot=0, figFilenamePattern=None):
    figFilename = figFilenamePattern.format(trialToPlot)
    nLatents = mMuK.shape[2]
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

