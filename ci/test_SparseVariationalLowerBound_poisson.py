
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

def test_eval():
    tol = 1e-5
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/Estep_Objective_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatent = len(mat['q_mu'])
    nTrials = mat['q_mu'][0,0].shape[2]
    qMu = [torch.from_numpy(mat['q_mu'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSVec = [torch.from_numpy(mat['q_sqrt'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSDiag = [torch.from_numpy(mat['q_diag'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    Kzzi = [torch.from_numpy(mat['Kzzi'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    Kzz = [torch.from_numpy(mat['Kzz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    quadKtz = [torch.from_numpy(mat['quadKtz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    quadKtt_tmp = torch.from_numpy(mat['quadKtt']).type(torch.DoubleTensor)
    hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
    hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)
    Y = torch.from_numpy(mat['Y']).type(torch.DoubleTensor).permute(2,0,1) 
    binWidth = torch.from_numpy(mat['BinWidth'])
    obj = mat['obj'][0,0]

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
    lbEval = svlb.eval()

    assert(abs(lbEval+obj)<tol)
    
if __name__=='__main__':
    test_eval()
