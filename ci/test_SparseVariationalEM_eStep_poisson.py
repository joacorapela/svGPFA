
import sys
import os
import pdb
import math
from scipy.io import loadmat
import torch
from approxPosteriorForH import ApproxPosteriorForH
from inducingPointsPrior import InducingPointsPrior
from kernelMatricesStore import KernelMatricesStore
from expectedLogLikelihood import PoissonExpectedLogLikelihood
from klDivergence import KLDivergence
from sparseVariationalLowerBound import SparseVariationalLowerBound
from sparseVariationalEM import SparseVariationalEM

def test_eval():
    tol = 1e-5
    verbose = True
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/Estep_Update_all_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatent = mat['q_mu'].shape[1]
    nTrials = mat['q_mu'][0,0].shape[2]
    qMu = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSVec = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSDiag = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    Kzzi = [torch.from_numpy(mat['Kzzi'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    Kzz = [torch.from_numpy(mat['Kzz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    quadKtz = [torch.from_numpy(mat['quadKtz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    quadKtt_tmp = torch.from_numpy(mat['quadKtt']).type(torch.DoubleTensor)
    hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
    hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)
    Y = torch.from_numpy(mat['Y']).type(torch.DoubleTensor).permute(2,0,1) 
    binWidth = mat['BinWidth'][0][0]
    maxIter = mat['maxIter'][0][0]
    nLowerBound = mat['nLowerBound'][0,0]

    quadKtt = torch.einsum('ij,k->kij', quadKtt_tmp, torch.ones(nTrials, dtype=torch.double))
    linkFunction = torch.exp

    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
    kernelMatricesStore = KernelMatricesStore(Kzz=Kzz, Kzzi=Kzzi, quadKtz=quadKtz, quadKtt=quadKtt)
    qH = ApproxPosteriorForH(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore)

    eLL = PoissonExpectedLogLikelihood(approxPosteriorForH=qH,
                                        hermQuadPoints=hermQuadPoints,
                                        hermQuadWeights=hermQuadWeights, 
                                        linkFunction=linkFunction, Y=Y,
                                        binWidth=binWidth)

    klDiv = KLDivergence(Kzzi=Kzzi, inducingPointsPrior=qU)
    svlb = SparseVariationalLowerBound(eLL=eLL, klDiv=klDiv)
    svEM = SparseVariationalEM(lowerBound=svlb, eLL=eLL, covMatricesStore=covMatricesStore)
    res = svEM._SparseVariationalEM__eStep(maxNIter=500, tol=1e-3, lr=1e-3, verbose=True)

    assert(res["lowerBound"]-(-nLowerBound)>0)

    pdb.set_trace()

if __name__=='__main__':
    test_eval()
