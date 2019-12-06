
import sys
import os
import pdb
from scipy.io import loadmat
import torch
import numpy as np
import plotUtils
from kernels import PeriodicKernel, ExponentialQuadraticKernel
import svGPFAModelFactory
from svEM import SVEM

def main(argv):
    figFilename = os.path.expanduser("figures/trueAndEstimatedLatents.png")
    yNonStackedFilename = os.path.expanduser("data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), 
                                "data/demo_PointProcess.mat")
    modelSaveFilename = os.path.join(os.path.dirname(__file__),
                                     "results/estimatedsvGPFAModelDeterministicLatents.pickle")
    lowerBoundHistFigFilename = os.path.join(os.path.dirname(__file__), "figures/lowerBoundHistDeterministicLatents.png")
    latentsFigFilename = os.path.join(os.path.dirname(__file__), "figures/estimatedDeterministicLatents.png")

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


    dataMat = loadmat(dataFilename)
    nLatents = len(dataMat['Z0'])
    nTrials = dataMat['Z0'][0,0].shape[2]
    qMu0 = [torch.from_numpy(dataMat['q_mu0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(dataMat['q_sqrt0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(dataMat['q_diag0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Z0 = [torch.from_numpy(dataMat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(dataMat["C0"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(dataMat["b0"]).type(torch.DoubleTensor).squeeze()
    legQuadPoints = torch.from_numpy(dataMat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(dataMat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    kernelNames = dataMat["kernelNames"]
    hprs0 = dataMat["hprs0"]
    testTimes = torch.from_numpy(dataMat['testTimes']).type(torch.DoubleTensor).squeeze()

    trueLatents = [[[] for k in range(nLatents)] for r in range(nTrials)]
    for r in range(nTrials):
        for k in range(nLatents):
            trueLatents[r][k] = {"mean": torch.from_numpy(dataMat['trueLatents'][r,k]).type(torch.DoubleTensor),
                                 "std": torch.zeros(dataMat['trueLatents'][r,k].shape)}

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = YNonStacked_tmp[r,0][n,0][:,0]

    kernels = [[None] for k in range(nLatents)]
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
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
    optimParams = {"emMaxNIter":20, "eStepMaxNIter":100, "mStepModelParamsMaxNIter":100, "mStepKernelParamsMaxNIter":100, "mStepKernelParamsLR":1e-5, "mStepIndPointsMaxNIter":100}

    model = svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=svGPFAModelFactory.PointProcess, 
        linkFunction=svGPFAModelFactory.ExponentialLink,
        embeddingType=svGPFAModelFactory.LinearEmbedding)

    svEM = SVEM()
    lowerBoundHist = svEM.maximize(model=model, measurements=YNonStacked, 
                                   kernels=kernels, initialParams=initialParams,
                                   quadParams=quadParams, 
                                   optimParams=optimParams)
    resultsToSave = {"lowerBoundHist": lowerBoundHist, "model": model}
    with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)
    testMuK, testVarK = model.predictLatents(newTimes=testTimes)
    indPointsLocs = model.getIndPointsLocs()

    plotUtils.plotLowerBoundHist(lowerBoundHist=lowerBoundHist, figFilename=lowerBoundHistFigFilename)
    plotUtils.plotTrueAndEstimatedLatents(times=testTimes, muK=testMuK, varK=testVarK, indPointsLocs=indPointsLocs, trueLatents=trueLatents, figFilename=latentsFigFilename)

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
