
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
import torch
from kernels import PeriodicKernel, ExponentialQuadraticKernel
from kernelMatricesStore import KernelMatricesStore
from approxPosteriorForH import ApproxPosteriorForHForAllNeuronsAllTimes, ApproxPosteriorForHForAllNeuronsAssociatedTimes
from inducingPointsPrior import InducingPointsPrior
from expectedLogLikelihood import PointProcessExpectedLogLikelihood, PoissonExpectedLogLikelihood
from klDivergence import KLDivergence
from sparseVariationalLowerBound import SparseVariationalLowerBound

def test_eval_pointProcess():
    tol = 3e-4
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    t = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    Z = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Y = [torch.from_numpy(mat['Y'][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    index = [torch.from_numpy(mat['index'][i,0][:,0]).type(torch.ByteTensor) for i in range(nTrials)]
    hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
    hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    obj = mat['obj'][0,0]

    linkFunction = torch.exp

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

    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
    kernelMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)

    qH_allNeuronsAllTimes = ApproxPosteriorForHForAllNeuronsAllTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore)
    qH_allNeuronsAssociatedTimes = ApproxPosteriorForHForAllNeuronsAssociatedTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore, neuronForSpikeIndex=index)

    eLL = PointProcessExpectedLogLikelihood(approxPosteriorForHForAllNeuronsAllTimes=qH_allNeuronsAllTimes, approxPosteriorForHForAllNeuronsAssociatedTimes=qH_allNeuronsAssociatedTimes, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, legQuadPoints=legQuadPoints, legQuadWeights=legQuadWeights, linkFunction=linkFunction)
    klDiv = KLDivergence(kernelMatricesStore=kernelMatricesStore, inducingPointsPrior=qU)
    svlb = SparseVariationalLowerBound(eLL=eLL, klDiv=klDiv)

    lbEval = svlb.eval()

    assert(abs(lbEval+obj)<tol)

    # pdb.set_trace()

def test_eval_poisson():
    tol = 1e-5
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['q_mu'])
    nTrials = mat['q_mu'][0,0].shape[2]
    qMu = [torch.from_numpy(mat['q_mu'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec = [torch.from_numpy(mat['q_sqrt'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag = [torch.from_numpy(mat['q_diag'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    t_tmp = torch.from_numpy(mat['tt']).type(torch.DoubleTensor).squeeze()
    Z = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Y = torch.from_numpy(mat['Y']).type(torch.DoubleTensor).permute(2,0,1) 
    C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
    hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)
    binWidth = torch.from_numpy(mat['BinWidth'])
    obj = mat['obj'][0,0]

    # t_tmp \in nQuad and we want t \in nTrials x nQuad x 1
    t = torch.ger(input=torch.ones(nTrials, dtype=torch.double), vec2=t_tmp).unsqueeze(dim=2)

    linkFunction = torch.exp

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

    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
    kernelMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
    qH = ApproxPosteriorForHForAllNeuronsAllTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore)

    eLL = PoissonExpectedLogLikelihood(approxPosteriorForHForAllNeuronsAllTimes=qH, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, linkFunction=linkFunction, Y=Y, binWidth=binWidth)
    klDiv = KLDivergence(kernelMatricesStore=kernelMatricesStore, inducingPointsPrior=qU)
    svlb = SparseVariationalLowerBound(eLL=eLL, klDiv=klDiv)
    lbEval = svlb.eval()

    assert(abs(lbEval+obj)<tol)
    
if __name__=='__main__':
    test_eval_pointProcess()
    test_eval_poisson()
