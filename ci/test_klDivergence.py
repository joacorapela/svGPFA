
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
import torch
from kernels import PeriodicKernel, ExponentialQuadraticKernel
from kernelMatricesStore import KernelMatricesStore
from klDivergence import KLDivergence
from inducingPointsPrior import InducingPointsPrior

def test_buildQSigma():
    tol = 1e-5
    dataFilename = os.path.expanduser('data/get_full_from_lowplusdiag.mat')

    mat = loadmat(dataFilename)
    nLatents = mat['q_sqrt'].shape[1]
    q_sqrt = [torch.from_numpy(mat['q_sqrt'][(0,i)]).permute(2,0,1) for i in range(nLatents)]
    q_diag = [torch.from_numpy(mat['q_diag'][(0,i)]).permute(2,0,1) for i in range(nLatents)]
    q_sigma = [torch.from_numpy(mat['q_sigma'][(0,i)]).permute(2,0,1) for i in range(nLatents)]

    svLowerBound = SparseVariationalLowerBound(eLL=None, covMatricesStore=None, qMu=None, qSVec=None, qSDiag=None, C=None, d=None, kernelParams=None, varRnk=torch.ones(3, dtype=torch.uint8), neuronForSpikeIndex=None)
    qSigma = svLowerBound._SparseVariationalLowerBound__buildQSigma(qSVec=q_sqrt, qSDiag=q_diag)

    error = torch.tensor([(qSigma[k]-q_sigma[k]).norm() for k in range(len(qSigma))]).sum()

    assert(error<tol)

def test_evalSumAcrossLatentsTrials():
    tol = 1e-5
    dataFilename = os.path.expanduser("data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['q_sqrt'].shape[1]
    nTrials = mat['Z'][0,0].shape[2]
    Kzzi = [torch.from_numpy(mat['Kzzi'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qMu = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    t = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    Z = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Y = [torch.from_numpy(mat['Y'][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    matKLDiv = torch.from_numpy(mat['KLd'])
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    kernels = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], 'PeriodicKernel'):
            kernels[k] = PeriodicKernel(scale=1.0, lengthScale=float(hprs[k,0][0]), period=float(hprs[k,0][1]))
        elif np.char.equal(kernelNames[0,k][0], 'rbfKernel'):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0, lengthScale=float(hprs[k,0][0]))
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    kernelMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
    klDiv = KLDivergence(kernelMatricesStore=kernelMatricesStore, inducingPointsPrior=qU)
    klDivEval = klDiv.evalSumAcrossLatentsAndTrials()

    klError = abs(matKLDiv-klDivEval)

    assert(klError<tol)

if __name__=="__main__":
    # test_buildQSigma()
    test_evalSumAcrossLatentsTrials()
