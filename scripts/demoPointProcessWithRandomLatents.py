
import sys
import os
import pdb
from scipy.io import loadmat
import torch
import numpy as np
import pickle
import plotUtils
from kernels import PeriodicKernel, ExponentialQuadraticKernel
import svGPFAModelFactory
from svEM import SVEM

def main(argv):
    k0Scale, k0LengthScale, k0Period = 0.1, 1.5, 1/2.5
    k1Scale, k1LengthScale, k1Period =.1, 1.2, 1/2.5,
    k2Scale, k2LengthScale = .1, 1
    latentsFilename = "results/latents_k0Scale{:.2f}_k0LengthScale{:.2f}_k0Period{:.2f}_k1Scale{:.2f}_k1LengthScale{:.2f}_k1Period{:.2f}_k2Scale{:.2f}_k2LengthScale{:.2f}.pickle".format(k0Scale, k0LengthScale, k0Period, k1Scale, k1LengthScale, k1Period, k2Scale, k2LengthScale)
    spikeTimesFilename = "results/spikeTimes_k0Scale{:.2f}_k0LengthScale{:.2f}_k0Period{:.2f}_k1Scale{:.2f}_k1LengthScale{:.2f}_k1Period{:.2f}_k2Scale{:.2f}_k2LengthScale{:.2f}.pickle".format(k0Scale, k0LengthScale, k0Period, k1Scale, k1LengthScale, k1Period, k2Scale, k2LengthScale)
    dataFilename = os.path.join(os.path.dirname(__file__),
                                "data/demo_PointProcess.mat")
    modelSaveFilename = os.path.join(os.path.dirname(__file__),
                                     "results/estimatedsvGPFAModel.pickle")
    figFilename = os.path.join(os.path.dirname(__file__),
                               "figures/estimatedLatents.png")

    with open(spikeTimesFilename, "rb") as f: spikeTimes = pickle.load(f)

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze()
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    kernelNames = mat["kernelNames"]
    hprs0 = mat["hprs0"]
    testTimes = torch.from_numpy(mat['testTimes']).type(torch.DoubleTensor).squeeze()
    trialToPlot = 2
    # trueLatents = [[torch.from_numpy(mat['trueLatents'][tr,k]).type(torch.DoubleTensor) for tr in range(nTrials)] for k in range(nLatents)]

    kernels = [[None] for k in range(nLatents)]
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        # kernels[k] = ExponentialQuadraticKernel()
        # kernelsParams0[k] = torch.tensor([1e-3,.1])
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = PeriodicKernel()
            kernelsParams0[k] = torch.tensor([1.0,
                                              float(hprs0[k,0][0]),
                                              float(hprs0[k,0][1])],
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = ExponentialQuadraticKernel()
            kernelsParams0[k] = torch.tensor([1.0,
                                              float(hprs0[k,0][0])],
                                             dtype=torch.double)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    qHParams0 = {"C0": C0, "d0": b0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    initialParams = {"svPosteriorOnIndPoints": qUParams0,
                     "kernelsMatricesStore": kmsParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}
    optimParams = {"emMaxNIter":100, "eStepMaxNIter":100, "mStepModelParamsMaxNIter":100, "mStepKernelParamsMaxNIter":100, "mStepKernelParamsLR":1e-5, "mStepIndPointsMaxNIter":100}

    model = svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=svGPFAModelFactory.PointProcess,
        linkFunction=svGPFAModelFactory.ExponentialLink,
        embeddingType=svGPFAModelFactory.LinearEmbedding)

    svEM = SVEM()
    lowerboundHist = svEM.maximize(model=model, measurements=spikeTimes,
                                   kernels=kernels, initialParams=initialParams,
                            quadParams=quadParams, optimParams=optimParams)
    resultsToSave = {"lowerBoundHist": lowerboundHist, "model": model}
    with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)
    with open(latentsFilename, "rb") as f: trueLatentsSamples = pickle.load( f)

    testMuK, testVarK = model.predictLatents(newTimes=testTimes)
    indPointsLocs = model.getIndPointsLocs()
    plotUtils.plotTrueAndEstimatedLatents(times=testTimes, muK=testMuK, varK=testVarK, indPointsLocs=indPointsLocs, trueLatents=trueLatentsSamples, trialToPlot=trialToPlot, figFilename=figFilename)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
