
import sys
import os
import pdb
import math
from scipy.io import loadmat
import torch
from klDivergence import KLDivergence
from inducingPointsPrior import InducingPointsPrior

def test_buildQSigma():
    tol = 1e-5
    dataFilename = os.path.expanduser('~/dev/research/gatsby/svGPFA/code/test/data/get_full_from_lowplusdiag.mat')

    mat = loadmat(dataFilename)
    nLatent = mat['q_sqrt'].shape[1]
    q_sqrt = [torch.from_numpy(mat['q_sqrt'][(0,i)]).permute(2,0,1) for i in range(nLatent)]
    q_diag = [torch.from_numpy(mat['q_diag'][(0,i)]).permute(2,0,1) for i in range(nLatent)]
    q_sigma = [torch.from_numpy(mat['q_sigma'][(0,i)]).permute(2,0,1) for i in range(nLatent)]

    svLowerBound = SparseVariationalLowerBound(eLL=None, covMatricesStore=None, qMu=None, qSVec=None, qSDiag=None, C=None, d=None, kernelParams=None, varRnk=torch.ones(3, dtype=torch.uint8), neuronForSpikeIndex=None)
    qSigma = svLowerBound._SparseVariationalLowerBound__buildQSigma(qSVec=q_sqrt, qSDiag=q_diag)

    error = torch.tensor([(qSigma[k]-q_sigma[k]).norm() for k in range(len(qSigma))]).sum()

    assert(error<tol)

def test_evalSumAcrossLatentsTrials():
    tol = 1e-5
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatent = mat['Kzzi'].shape[0]
    Kzzi = [torch.from_numpy(mat['Kzzi'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qMu = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSVec = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSDiag = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    matKLDiv = torch.from_numpy(mat['KLd'])

    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
    klDiv = KLDivergence(Kzzi=Kzzi, inducingPointsPrior=qU)
    klDivEval = klDiv.evalSumAcrossLatentsAndTrials()

    klError = abs(matKLDiv-klDivEval)

    assert(klError<tol)

if __name__=="__main__":
    # test_buildQSigma()
    test_evalSumAcrossLatentsTrials()
