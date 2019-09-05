
import sys
import os
import pdb
import math
from scipy.io import loadmat
import torch
from inducingPointsPrior import InducingPointsPrior
from covarianceMatricesStore import PointProcessCovarianceMatricesStore
from approxPosteriorForH import PointProcessApproxPosteriorForH
from klDivergence import KLDivergence
from expectedLogLikelihood import PointProcessExpectedLogLikelihood
from sparseVariationalLowerBound import SparseVariationalLowerBound
from sparseVariationalEM import SparseVariationalEM

def test_eval():
    tol = 1e-5
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatent = mat['spikeKtz'].shape[0]
    nTrials = mat['spikeKtz'].shape[1]
    qMu = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSVec = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSDiag = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    Kzzi = [torch.from_numpy(mat['Kzzi'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    Kzz = [torch.from_numpy(mat['Kzz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    quadKtz = [torch.from_numpy(mat['quadKtz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    quadKtt = torch.from_numpy(mat['quadKtt']).type(torch.DoubleTensor).permute(2,0,1)
    spikeKtz = [[torch.from_numpy(mat['spikeKtz'][k,tr]).type(torch.DoubleTensor) for tr in range(nTrials)] for k in range(nLatent)]
    spikeKtt = [[torch.from_numpy(mat['spikeKtt'][k,tr]).type(torch.DoubleTensor) for tr in range(nTrials)] for k in range(nLatent)]
    mu_h_spikes = [torch.from_numpy(mat['mu_h_Spikes'][0,i]).type(torch.DoubleTensor).squeeze() for i in range(nTrials)]
    var_h_spikes = [torch.from_numpy(mat['var_h_Spikes'][0,i]).type(torch.DoubleTensor).squeeze() for i in range(nTrials)]
    index = [torch.from_numpy(mat['index'][i,0][:,0]).type(torch.ByteTensor) for i in range(nTrials)]
    obj = mat['obj'][0,0]

    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
    hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)

    linkFunction = torch.exp

    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
    covMatricesStore = PointProcessCovarianceMatricesStore(Kzz=Kzz, Kzzi=Kzzi, quadKtz=quadKtz, quadKtt=quadKtt, spikeKtz=spikeKtz, spikeKtt=spikeKtt)
    qH = PointProcessApproxPosteriorForH(C=C, d=b, inducingPointsPrior=qU, covMatricesStore=covMatricesStore, neuronForSpikeIndex=index)

    eLL = PointProcessExpectedLogLikelihood(approxPosteriorForH=qH,
                                             hermQuadPoints=hermQuadPoints, 
                                             hermQuadWeights=hermQuadWeights, 
                                             legQuadPoints=legQuadPoints,
                                             legQuadWeights=legQuadWeights, 
                                             linkFunction=linkFunction)
    klDiv = KLDivergence(Kzzi=Kzzi, inducingPointsPrior=qU)
    svLB = SparseVariationalLowerBound(eLL=eLL, klDiv=klDiv)
    svEM = SparseVariationalEM(lowerBound=svLB)
    maxRes = svEM.maximize()

    pdb.set_trace()

    assert(False)
    
if __name__=='__main__':
    test_eval()
