
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
import torch
from svPosteriorOnIndPoints import SVPosteriorOnIndPoints
from kernels import PeriodicKernel, ExponentialQuadraticKernel
from kernelMatricesStore import IndPointsLocsKMS
from klDivergence import KLDivergence

def test_evalSumAcrossLatentsTrials():
    tol = 1e-5
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['q_sqrt'].shape[1]
    nTrials = mat['Z'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    matKLDiv = torch.from_numpy(mat['KLd'])
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    kernels = [[None] for k in range(nLatents)]
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]), 
                                              float(hprs[k,0][1])], 
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
                                             dtype=torch.double)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}

    indPointsLocsKMS = IndPointsLocsKMS()
    qU = SVPosteriorOnIndPoints()
    klDiv = KLDivergence(indPointsLocsKMS=indPointsLocsKMS, 
                         svPosteriorOnIndPoints=qU)

    qU.setInitialParams(initialParams=qUParams0)
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setInitialParams(initialParams=kmsParams0)
    indPointsLocsKMS.buildKernelsMatrices()
    klDivEval = klDiv.evalSumAcrossLatentsAndTrials()

    klError = abs(matKLDiv-klDivEval)

    assert(klError<tol)

if __name__=="__main__":
    test_evalSumAcrossLatentsTrials()
