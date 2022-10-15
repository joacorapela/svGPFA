import sys
import os
from scipy.io import loadmat
import torch
import svGPFA.stats.svPosteriorOnIndPoints
import svGPFA.utils.miscUtils

# def test_get_flattened_params():
#     nTrials = 2
#     nIndPoints = [2, 2, 2]
# 
#     nLatents = len(nIndPoints)
#     qMu0 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
#     srQSigma0Vecs = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
#     initial_params = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
# 
#     true_flattened_params = []
#     for k in range(nLatents):
#         true_flattened_params.extend(qMu0[k].flatten().tolist())
#     for k in range(nLatents):
#         true_flattened_params.extend(srQSigma0Vecs[k].flatten().tolist())
# 
#     svPosteriorOnIndPoints = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
#     svPosteriorOnIndPoints.setInitialParams(initial_params=initial_params)
#     flattened_params = svPosteriorOnIndPoints.get_flattened_params()
# 
#     assert(flattened_params==true_flattened_params)
# 
# def test_set_flattened_params():
#     nTrials = 2
#     nIndPoints = [2, 2, 2]
# 
#     nLatents = len(nIndPoints)
# 
#     qMu0_1 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
#     srQSigma0Vecs_1 = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
#     initial_params_1 = {"qMu0": qMu0_1, "srQSigma0Vecs": srQSigma0Vecs_1}
#     svPosteriorOnIndPoints_1 = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
#     svPosteriorOnIndPoints_1.setInitialParams(initial_params=initial_params_1)
#     flattened_params_1 = svPosteriorOnIndPoints_1.get_flattened_params()
# 
#     qMu0_2 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
#     srQSigma0Vecs_2 = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
#     initial_params_2 = {"qMu0": qMu0_2, "srQSigma0Vecs": srQSigma0Vecs_2}
#     svPosteriorOnIndPoints_2 = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
#     svPosteriorOnIndPoints_2.setInitialParams(initial_params=initial_params_2)
#     svPosteriorOnIndPoints_2.set_params_from_flattened(flattened_params=flattened_params_1)
#     flattened_params_2 = svPosteriorOnIndPoints_2.get_flattened_params()
# 
#     assert(flattened_params_1==flattened_params_2)
# 
# def test_set_params_requires_grad():
#     nTrials = 2
#     nIndPoints = [2, 2, 2]
# 
#     nLatents = len(nIndPoints)
# 
#     qMu0 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
#     srQSigma0Vecs = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
#     initial_params = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
#     svPosteriorOnIndPoints = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
#     svPosteriorOnIndPoints.setInitialParams(initial_params=initial_params)
#     svPosteriorOnIndPoints.set_params_requires_grad(requires_grad=True)
#     params = svPosteriorOnIndPoints.getParams()
#     for param in params:
#         assert(param.requires_grad)
# 
#     svPosteriorOnIndPoints.set_params_requires_grad(requires_grad=False)
#     params = svPosteriorOnIndPoints.getParams()
#     for param in params:
#         assert(not param.requires_grad)

def test_buildCov():
    tol = 1e-5
    dataFilename = os.path.join(os.path.dirname(__file__), "data/get_full_from_lowplusdiag.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['q_sqrt'].shape[0]
    nTrials = mat['q_sqrt'][(0,0)].shape[2]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    q_sigma = [torch.from_numpy(mat['q_sigma'][(0,k)]).permute(2,0,1) for k in range(nLatents)]
    qMu0 = [[] for i in range(nLatents)]

    params0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
    qU = svGPFA.stats.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
    qU.setInitialParams(initial_params=params0)
    qSigma = qU.buildCov();

    error = torch.tensor([(qSigma[k]-q_sigma[k]).norm() for k in range(len(qSigma))]).sum()

    assert(error<tol)

if __name__=="__main__":
    test_buildCov()
