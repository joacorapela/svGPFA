
import sys
import os
import torch
import pdb
import pickle
import matplotlib.pyplot as plt
from scipy.io import loadmat
sys.path.append("../src")

def main(argv):
    marker = 'x'
    pModelSaveFilename = "results/estimationResLeasSimulation.pickle"
    mLowerBoundFilename = "../../matlabCode/scripts/results/lowerBound.mat"
    lowerBoundVsIterNo = "figures/lowerBoundVsIterNo.png"
    lowerBoundVsElapsedTime = "figures/lowerBoundVsRuntime.png"

    with open(pModelSaveFilename, "rb") as f: res = pickle.load(f)
    pLowerBound = -torch.stack(res["lowerBoundHist"]).detach().numpy()
    pElapsedTime = res["elapsedTimeHist"]

    loadRes = loadmat(mLowerBoundFilename)
    mLowerBound = loadRes["lowerBound"]
    mElapsedTime = loadRes["elapsedTime"]

    plt.plot(pLowerBound, marker=marker, label="Python")
    plt.plot(mLowerBound, marker=marker, label="Matlab")
    plt.legend()
    plt.xlabel("Iteration Number")
    plt.ylabel("Lower Bound")
    plt.savefig(lowerBoundVsIterNo)

    plt.figure()
    plt.plot(pElapsedTime, pLowerBound, marker=marker, label="Python")
    plt.plot(mElapsedTime, mLowerBound, marker=marker, label="Matlab")
    plt.legend()
    plt.xlabel("Elapsed Time (sec)")
    plt.ylabel("Lower Bound")
    plt.savefig(lowerBoundVsElapsedTime)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
