
import sys
import pdb
import os
import math
from scipy.io import loadmat
import numpy as np
import torch
sys.path.append("../src")
from stats.kernels import PeriodicKernel, ExponentialQuadraticKernel
from stats.svGPFA.kernelMatricesStore import IndPointsLocsKMS, \
        IndPointsLocsAndAllTimesKMS, IndPointsLocsAndAssocTimesKMS
from stats.svGPFA.svPosteriorOnIndPoints import SVPosteriorOnIndPoints
from stats.svGPFA.svPosteriorOnLatents import SVPosteriorOnLatentsAllTimes,\
        SVPosteriorOnLatentsAssocTimes
from stats.svGPFA.svEmbedding import LinearSVEmbeddingAllTimes, \
        LinearSVEmbeddingAssocTimes

def test_computeMeansAndVars_allTimes():
    tol = 5e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat["Z"].shape[0]
    nTrials = mat["Z"][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat["q_mu"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat["q_sqrt"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat["q_diag"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    t = torch.from_numpy(mat["ttQuad"]).type(torch.DoubleTensor).permute(2, 0, 1)
    Z0 = [torch.from_numpy(mat["Z"][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Y = [torch.from_numpy(mat["Y"][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor)
    mu_h = torch.from_numpy(mat["mu_h_Quad"]).type(torch.DoubleTensor).permute(2,0,1)
    var_h = torch.from_numpy(mat["var_h_Quad"]).type(torch.DoubleTensor).permute(2,0,1)
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

    qU = SVPosteriorOnIndPoints()
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
    qK = SVPosteriorOnLatentsAllTimes(svPosteriorOnIndPoints=qU, 
                                      indPointsLocsKMS=indPointsLocsKMS, 
                                      indPointsLocsAndTimesKMS=
                                       indPointsLocsAndAllTimesKMS)
    qH = LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qK)
    qH.setKernels(kernels=kernels)

    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    qHParams0 = {"C0": C0, "d0": b0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    initialParams = {"svPosteriorOnIndPoints": qUParams0,
                     "kernelsMatricesStore": kmsParams0,
                     "svEmbedding": qHParams0}
    qH.setInitialParams(initialParams=initialParams)
    qH.setTimes(times=t)
    qH.setIndPointsLocsKMSEpsilon(indPointsLocsKMSEpsilon=1e-5) # Fix: need to read indPointsLocsKMSEpsilon from Matlab's CI test data
    qH.buildKernelsMatrices()
    qHMu, qHVar = qH.computeMeansAndVars()

    qHMuError = math.sqrt(((mu_h-qHMu)**2).mean())
    assert(qHMuError<tol)
    qHVarError = math.sqrt(((var_h-qHVar)**2).mean())
    assert(qHVarError<tol)

def test_computeMeansAndVars_assocTimes():
    tol = 5e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat["Z"].shape[0]
    nTrials = mat["Z"][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat["q_mu"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat["q_sqrt"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat["q_diag"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    t = torch.from_numpy(mat["ttQuad"]).type(torch.DoubleTensor).permute(2, 0, 1)
    Z0 = [torch.from_numpy(mat["Z"][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Y = [torch.from_numpy(mat["Y"][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor)
    mu_h = [torch.from_numpy(mat["mu_h_Spikes"][0,i]).type(torch.DoubleTensor).squeeze() for i in range(nTrials)]
    var_h = [torch.from_numpy(mat["var_h_Spikes"][0,i]).type(torch.DoubleTensor).squeeze() for i in range(nTrials)]
    index = [torch.from_numpy(mat["index"][i,0][:,0]).type(torch.ByteTensor)-1 for i in range(nTrials)]

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

    qU = SVPosteriorOnIndPoints()
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsAndAssocTimesKMS = IndPointsLocsAndAssocTimesKMS()
    qK = SVPosteriorOnLatentsAssocTimes(svPosteriorOnIndPoints=qU, 
                                        indPointsLocsKMS=indPointsLocsKMS, 
                                        indPointsLocsAndTimesKMS=
                                         indPointsLocsAndAssocTimesKMS)
    qH = LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=qK)
    qH.setKernels(kernels=kernels)

    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    qHParams0 = {"C0": C0, "d0": b0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    initialParams = {"svPosteriorOnIndPoints": qUParams0,
                     "kernelsMatricesStore": kmsParams0,
                     "svEmbedding": qHParams0}
    qH.setInitialParams(initialParams=initialParams)
    qH.setKernels(kernels=kernels)
    qH.setTimes(times=Y)
    qH.setNeuronForSpikeIndex(neuronForSpikeIndex=index)

    # begin patches because we are not using SVPosteriorOnLatentsAssocTimes in 
    # conjunction with SVPosteriorOnLatentsAllTimes
    qU.setInitialParams(initialParams=qUParams0)
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setInitialParams(initialParams=kmsParams0)
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setEpsilon(epsilon=1e-5) # Fix: need to read indPointsLocsKMSEpsilon from Matlab's CI test data
    indPointsLocsKMS.buildKernelsMatrices()
    # end patches because we are not using SVPosteriorOnLatentsAssocTimes in 
    # conjunction with SVPosteriorOnLatentsAllTimes

    qH.buildKernelsMatrices()
    qHMu, qHVar = qH.computeMeansAndVars()

    for i in range(len(mu_h)):
        qHMuError = math.sqrt(torch.sum((mu_h[i]-qHMu[i])**2))/mu_h[i].shape[0]
        assert(qHMuError<tol)
        qHVarError = math.sqrt(torch.sum((var_h[i]-qHVar[i])**2))/\
                     var_h[i].shape[0]
        assert(qHVarError<tol)
   
if __name__=="__main__":
    test_computeMeansAndVars_allTimes()
    test_computeMeansAndVars_assocTimes()
