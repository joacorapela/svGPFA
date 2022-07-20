import pdb
import sys
import os
from scipy.io import loadmat
import torch
sys.path.append("../src")
import stats.svGPFA.svPosteriorOnIndPoints
import utils.svGPFA.miscUtils

# def test_get_flattened_params():
#     nTrials = 2
#     nIndPoints = [2, 2, 2]
# 
#     nLatents = len(nIndPoints)
#     qMu0 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
#     srQSigma0Vecs = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
#     initialParams = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
# 
#     true_flattened_params = []
#     for k in range(nLatents):
#         true_flattened_params.extend(qMu0[k].flatten().tolist())
#     for k in range(nLatents):
#         true_flattened_params.extend(srQSigma0Vecs[k].flatten().tolist())
# 
#     svPosteriorOnIndPoints = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
#     svPosteriorOnIndPoints.setInitialParams(initialParams=initialParams)
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
#     initialParams_1 = {"qMu0": qMu0_1, "srQSigma0Vecs": srQSigma0Vecs_1}
#     svPosteriorOnIndPoints_1 = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
#     svPosteriorOnIndPoints_1.setInitialParams(initialParams=initialParams_1)
#     flattened_params_1 = svPosteriorOnIndPoints_1.get_flattened_params()
# 
#     qMu0_2 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
#     srQSigma0Vecs_2 = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
#     initialParams_2 = {"qMu0": qMu0_2, "srQSigma0Vecs": srQSigma0Vecs_2}
#     svPosteriorOnIndPoints_2 = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
#     svPosteriorOnIndPoints_2.setInitialParams(initialParams=initialParams_2)
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
#     initialParams = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
#     svPosteriorOnIndPoints = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
#     svPosteriorOnIndPoints.setInitialParams(initialParams=initialParams)
#     svPosteriorOnIndPoints.set_params_requires_grad(requires_grad=True)
#     params = svPosteriorOnIndPoints.getParams()
#     for param in params:
#         assert(param.requires_grad)
# 
#     svPosteriorOnIndPoints.set_params_requires_grad(requires_grad=False)
#     params = svPosteriorOnIndPoints.getParams()
#     for param in params:
#         assert(not param.requires_grad)

def test_buildQSigma():
    tol = 1e-5
    dataFilename = os.path.join(os.path.dirname(__file__), "data/get_full_from_lowplusdiag.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['q_sqrt'].shape[0]
    nTrials = mat['q_sqrt'][(0,0)].shape[2]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    srQSigma0Vecs = utils.svGPFA.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    q_sigma = [torch.from_numpy(mat['q_sigma'][(0,k)]).permute(2,0,1) for k in range(nLatents)]
    qMu0 = [[] for i in range(nLatents)]

    params0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
    qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
    qU.setInitialParams(initialParams=params0)
    qSigma = qU.buildQSigma();

    error = torch.tensor([(qSigma[k]-q_sigma[k]).norm() for k in range(len(qSigma))]).sum()

    assert(error<tol)

if __name__=="__main__":
    test_buildQSigma()
