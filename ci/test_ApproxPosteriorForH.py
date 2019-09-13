
import sys
import pdb
import os
import math
from scipy.io import loadmat
import torch
from approxPosteriorForH import ApproxPosteriorForH
from inducingPointsPrior import InducingPointsPrior
from kernelMatricesStore import PointProcessKernelMatricesStore

def test_getMeanAndVarianceAtQuadPoints():
    tol = 1e-6
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatent = mat['spikeKtz'].shape[0]
    nTrials = mat['spikeKtz'].shape[1]
    qMu = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSVec = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSDiag = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor)
    Kzzi = [torch.from_numpy(mat['Kzzi'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    Kzz = [torch.from_numpy(mat['Kzz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    quadKtz = [torch.from_numpy(mat['quadKtz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    quadKtt = torch.from_numpy(mat['quadKtt']).type(torch.DoubleTensor).permute(2,0,1)
    spikeKtz = [[torch.from_numpy(mat['spikeKtz'][k,tr]).type(torch.DoubleTensor) for tr in range(nTrials)] for k in range(nLatent)]
    spikeKtt = [[torch.from_numpy(mat['spikeKtt'][k,tr]).type(torch.DoubleTensor) for tr in range(nTrials)] for k in range(nLatent)]
    mu_h = torch.from_numpy(mat['mu_h_Quad']).type(torch.DoubleTensor).permute(2,0,1)
    var_h = torch.from_numpy(mat['var_h_Quad']).type(torch.DoubleTensor).permute(2,0,1)

    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
    covMatricesStore = PointProcessKernelMatricesStore(Kzz=Kzz, Kzzi=Kzzi, quadKtz=quadKtz, quadKtt=quadKtt, spikeKtz=spikeKtz, spikeKtt=spikeKtt)
    qH = ApproxPosteriorForH(C=C, d=b, inducingPointsPrior=qU, covMatricesStore=covMatricesStore)
    qHMu, qHVar = qH.getMeanAndVarianceAtQuadPoints()

    qHMuError = math.sqrt(((mu_h-qHMu)**2).mean())
    qHVarError = math.sqrt(((var_h-qHVar)**2).mean())

    assert(qHMuError<tol)
    assert(qHVarError<tol)

if __name__=="__main__":
    test_getMeanAndVarianceAtQuadPoints()
