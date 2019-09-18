import sys
import os
import pickle
from scipy.io import loadmat
import numpy as np
import torch
import matplotlib.pyplot as plt

def main(argv):
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/demos/data/demo_PointProcess.mat")
    approxPosteriorForH_allNeuronsAllTimes_pickleFilename = "data/approxPosteriorForH_allNeuronsAllTimes.pickle"

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]

    testTimes = torch.from_numpy(mat['testTimes']).type(torch.DoubleTensor).squeeze()
    trueLatents = [[torch.from_numpy(mat['trueLatents'][tr,k]).type(torch.DoubleTensor) for tr in range(nTrials)] for k in range(nLatents)]

    file = open(approxPosteriorForH_allNeuronsAllTimes_pickleFilename, "rb")
    qH_allNeuronsAllTimes = pickle.load(file)
    file.close()

    qHMu, qHVar, qKMu, qKVar = qH_allNeuronsAllTimes.predict(testTimes=testTimes)
    Z = qH_allNeuronsAllTimes._kernelMatricesStore.getZ()
    testTimesToPlot = testTimes.numpy()
    trialToPlot = 0
    f, axes = plt.subplots(nLatents, 1, sharex=True)
    for k in range(nLatents):
        trueLatentToPlot = trueLatents[k][trialToPlot].numpy().squeeze()
        qKMuToPlot = qKMu[trialToPlot,:,k].numpy()
        errorToPlot = qKVar[trialToPlot,:,k].sqrt().numpy()
        axes[k].plot(testTimesToPlot, trueLatentToPlot, label="true", color="black")
        axes[k].plot(testTimesToPlot, qKMuToPlot, label="estimated", color="blue")
        axes[k].fill_between(testTimesToPlot, qKMuToPlot-errorToPlot, qKMuToPlot+errorToPlot, color="lightblue")
        for i in range(Z[k].shape[1]):
            axes[k].axvline(x=Z[k][trialToPlot,i, 0], color="red")
        axes[k].set_ylabel("Latent %d"%(k))
    axes[-1].set_xlabel("Sample")
    axes[-1].legend()
    plt.xlim(left=np.min(testTimesToPlot)-1, right=np.max(testTimesToPlot)+1)
    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
