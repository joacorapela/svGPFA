import pdb
import sys
import os
from scipy.io import loadmat
import torch
sys.path.append("../src")
import stats.svGPFA.svPosteriorOnIndPoints
import utils.svGPFA.miscUtils

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
