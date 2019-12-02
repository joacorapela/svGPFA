
import matplotlib.pyplot as plt
import numpy as np

def plotTrueAndEstimatedLatents(times, muK, varK, indPointsLocs, trueLatents, trialToPlot=0, figFilename=None):
    nLatents = muK.shape[2]
    timesToPlot = times.numpy()
    f, axes = plt.subplots(nLatents, 1, sharex=True)
    for k in range(nLatents):
        trueLatentToPlot = trueLatents[k][trialToPlot].numpy().squeeze()
        muKToPlot = muK[trialToPlot,:,k].detach().numpy()
        errorToPlot = varK[trialToPlot,:,k].sqrt().detach().numpy()
        axes[k].plot(timesToPlot, trueLatentToPlot, label="true", color="black")
        axes[k].plot(timesToPlot, muKToPlot, label="estimated", color="blue")
        axes[k].fill_between(timesToPlot, muKToPlot-errorToPlot, 
                             muKToPlot+errorToPlot, color="lightblue")
        for i in range(indPointsLocs[k].shape[1]):
            axes[k].axvline(x=indPointsLocs[k][trialToPlot,i, 0], color="red")
        axes[k].set_ylabel("Latent %d"%(k))
    axes[-1].set_xlabel("Sample")
    axes[-1].legend()
    plt.xlim(left=np.min(timesToPlot)-1, right=np.max(timesToPlot)+1)
    if figFilename is not None:
        plt.savefig(fname=figFilename)
    plt.show()

