
import sys
import pdb
import os
import math
from scipy.io import loadmat
import numpy as np
import torch
import svGPFA.utils.miscUtils
import svGPFA.stats.kernels
import svGPFA.stats.kernelsMatricesStore
import svGPFA.stats.variationalDist
import svGPFA.stats.posteriorOnLatents

def test_computeMeansAndVars_quadTimes():
    tol = 5e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat["Z"].shape[0]
    nTrials = mat["Z"][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat["q_mu"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat["q_sqrt"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat["q_diag"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    t = torch.from_numpy(mat["ttQuad"]).type(torch.DoubleTensor).permute(2, 0, 1)
    Z0 = [torch.from_numpy(mat["Z"][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    mu_k = torch.from_numpy(mat["mu_k_Quad"]).type(torch.DoubleTensor).permute(2,0,1)
    var_k = torch.from_numpy(mat["var_k_Quad"]).type(torch.DoubleTensor).permute(2,0,1)
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    kernels = [[None] for k in range(nLatents)]
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]),
                                              float(hprs[k,0][1])],
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
                                             dtype=torch.double)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qU = svGPFA.stats.variationalDist.VariationalDistChol()
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
    indPointsLocsAndTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
    qK = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsQuadTimes(
        variationalDist=qU, indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndTimesKMS)

    qUParams0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
    kmsParams0 = {"kernels_params0": kernelsParams0,
                  "inducing_points_locs0": Z0}

    qU.setInitialParams(initial_params=qUParams0)

    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setInitialParams(initial_params=kmsParams0)
    indPointsLocsKMS.setRegParam(reg_param=1e-5) # Fix: need to read indPointsLocsKMSEpsilon from Matlab's CI test data
    indPointsLocsKMS.buildKernelsMatrices()

    indPointsLocsAndTimesKMS.setKernels(kernels=kernels)
    indPointsLocsAndTimesKMS.setInitialParams(initial_params=kmsParams0)
    indPointsLocsAndTimesKMS.setTimes(times=t)
    indPointsLocsAndTimesKMS.buildKernelsMatrices()

    qKMu, qKVar = qK.computeMeansAndVars()

    for r in range(len(qKMu)):
        qKMuError = math.sqrt(((mu_k[r,:,:] - qKMu[r])**2).mean())
        assert(qKMuError<tol)
        qKVarError = math.sqrt(((var_k[r,:,:]-qKVar[r])**2).mean())
        assert(qKVarError<tol)

def test_computeMeansAndVars_spikesTimes():
    tol = 5e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat["Z"].shape[0]
    nTrials = mat["Z"][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat["q_mu"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat["q_sqrt"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat["q_diag"][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [torch.from_numpy(mat["Z"][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Y = [torch.from_numpy(mat["Y"][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    mu_k = [torch.from_numpy(mat["mu_k_Spikes"][0,tr]).type(torch.DoubleTensor) for tr in range(nTrials)]
    var_k = [torch.from_numpy(mat["var_k_Spikes"][0,tr]).type(torch.DoubleTensor) for tr in range(nTrials)]
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    kernels = [[None] for k in range(nLatents)]
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]),
                                              float(hprs[k,0][1])],
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
                                             dtype=torch.double)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qU = svGPFA.stats.variationalDist.VariationalDistChol()
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
    indPointsLocsAndTimesKMS = \
        svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
    qK = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsSpikesTimes(
        variationalDist=qU, indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndTimesKMS)

    # qSigma0[k] \in nTrials x nInd[k] x nInd[k]
    qUParams0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
    kmsParams0 = {"kernels_params0": kernelsParams0,
                  "inducing_points_locs0": Z0}
    qU.setInitialParams(initial_params=qUParams0)

    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setInitialParams(initial_params=kmsParams0)
    indPointsLocsKMS.setRegParam(reg_param=1e-5) # Fix: need to read indPointsLocsKMSEpsilon from Matlab's CI test data
    indPointsLocsKMS.buildKernelsMatrices()

    indPointsLocsAndTimesKMS.setKernels(kernels=kernels)
    indPointsLocsAndTimesKMS.setInitialParams(initial_params=kmsParams0)
    indPointsLocsAndTimesKMS.setTimes(times=Y)
    indPointsLocsAndTimesKMS.buildKernelsMatrices()

    qKMu, qKVar = qK.computeMeansAndVars()

    for tr in range(nTrials):
        qKMuError = math.sqrt(((mu_k[tr]-qKMu[tr])**2).mean())
        assert(qKMuError<tol)
        qKVarError = math.sqrt(((var_k[tr]-qKVar[tr])**2).mean())
        assert(qKVarError<tol)

if __name__=="__main__":
    test_computeMeansAndVars_quadTimes()
    test_computeMeansAndVars_spikesTimes()
