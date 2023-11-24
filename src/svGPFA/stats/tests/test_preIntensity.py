
import sys
import os
import math
from scipy.io import loadmat
import numpy as np
import torch
import svGPFA.utils.miscUtils
import svGPFA.stats.kernels
import svGPFA.stats.kernelsMatricesStore
import svGPFA.stats.svPosteriorOnIndPoints
import svGPFA.stats.svPosteriorOnLatents
import svGPFA.stats.svEmbedding

def test_computeMeansAndVars_allTimes():
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

    qU = svGPFA.stats.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
    indPointsLocsAndAllTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
    qK = svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(svPosteriorOnIndPoints=qU, 
                                      indPointsLocsKMS=indPointsLocsKMS, 
                                      indPointsLocsAndTimesKMS=
                                       indPointsLocsAndAllTimesKMS)
    qH = svGPFA.stats.svEmbedding.LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qK)
    qH.setKernels(kernels=kernels)

    qUParams0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
    kmsParams0 = {"kernels_params0": kernelsParams0,
                  "inducing_points_locs0": Z0}
    qKParams0 = {"posterior_on_ind_points": qUParams0,
                 "kernels_matrices_store": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initial_params = {"posterior_on_latents": qKParams0,
                     "embedding": qHParams0}
    qH.setInitialParams(initial_params=initial_params)
    qH.setTimes(times=t)
    qH.setPriorCovRegParam(priorCovRegParam=1e-5) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
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
    srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
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

    qU = svGPFA.stats.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
    indPointsLocsAndAssocTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAssocTimesKMS()
    qK = svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes(svPosteriorOnIndPoints=qU, 
                                        indPointsLocsKMS=indPointsLocsKMS, 
                                        indPointsLocsAndTimesKMS=
                                         indPointsLocsAndAssocTimesKMS)
    qH = svGPFA.stats.svEmbedding.LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=qK)
    qH.setKernels(kernels=kernels)

    qUParams0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
    kmsParams0 = {"kernels_params0": kernelsParams0,
                  "inducing_points_locs0": Z0}
    qKParams0 = {"posterior_on_ind_points": qUParams0,
                 "kernels_matrices_store": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initial_params = {"posterior_on_latents": qKParams0,
                     "embedding": qHParams0}
    qH.setInitialParams(initial_params=initial_params)
    qH.setKernels(kernels=kernels)
    qH.setTimes(times=Y)
    qH.setNeuronForSpikeIndex(neuronForSpikeIndex=index)

    # begin patches because we are not using SVPosteriorOnLatentsAssocTimes in 
    # conjunction with SVPosteriorOnLatentsAllTimes
    qU.setInitialParams(initial_params=qUParams0)
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setInitialParams(initial_params=kmsParams0)
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setRegParam(reg_param=1e-5) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
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
