
import pdb
import torch
import matplotlib.pyplot as plt

def plotLowerBoundHist(lowerBoundHist, xlabel="Iteration Number", ylabel="Lower Bound", marker="x", linestyle="-", figFilename=None):
    plt.plot(lowerBoundHist, marker=marker, linestyle=linestyle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if figFilename is not None:
        plt.savefig(fname=figFilename)
    plt.show()

def plotTrueAndEstimatedLatents(times, muK, varK, indPointsLocs, trueLatents,
                                trialToPlot=0, figFilename=None):
    nLatents = muK.shape[2]
    timesToPlot = times
    f, axes = plt.subplots(nLatents, 1, sharex=True)
    title = "Trial {:d}".format(trialToPlot)
    axes[0].set_title(title)
    if figFilename is not None:
        plt.savefig(fname=figFilename)
    for k in range(nLatents):
        trueMeanToPlot = trueLatents[trialToPlot][k]["mean"].squeeze()
        trueCIToPlot = 1.96*(trueLatents[trialToPlot][k]["std"].squeeze())
        hatMeanToPlot = muK[trialToPlot,:,k]
        positiveMSE = torch.mean((trueMeanToPlot-hatMeanToPlot)**2)
        negativeMSE = torch.mean((trueMeanToPlot+hatMeanToPlot)**2)
        if negativeMSE<positiveMSE:
            hatMeanToPlot = -hatMeanToPlot
        hatCIToPlot = 1.96*(varK[trialToPlot,:,k].sqrt())
        axes[k].plot(timesToPlot.detach().numpy(), trueMeanToPlot, label="true", color="black")
        axes[k].fill_between(timesToPlot, trueMeanToPlot-trueCIToPlot, trueMeanToPlot+trueCIToPlot, color="lightgray")
        axes[k].plot(timesToPlot, hatMeanToPlot.detach().numpy(), label="estimated", color="blue")
        axes[k].fill_between(timesToPlot, (hatMeanToPlot-hatCIToPlot).detach().numpy(), (hatMeanToPlot+hatCIToPlot).detach().numpy(), color="lightblue")
        for i in range(indPointsLocs[k].shape[1]):
            axes[k].axvline(x=indPointsLocs[k][trialToPlot,i, 0], color="red")
        axes[k].set_ylabel("Latent %d"%(k))
    axes[-1].set_xlabel("Sample")
    axes[-1].legend()
    plt.xlim(left=torch.min(timesToPlot)-1, right=torch.max(timesToPlot)+1)
    plt.show()

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

