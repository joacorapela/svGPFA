
import sys
import os
from scipy.io import loadmat
import torch
from svPosteriorOnIndPoints import SVPosteriorOnIndPoints

def test_buildQSigma():
    tol = 1e-5
    dataFilename = os.path.join(os.path.dirname(__file__), "data/get_full_from_lowplusdiag.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['q_sqrt'].shape[1]
    q_sqrt = [torch.from_numpy(mat['q_sqrt'][(0,i)]).permute(2,0,1) for i in range(nLatents)]
    q_diag = [torch.from_numpy(mat['q_diag'][(0,i)]).permute(2,0,1) for i in range(nLatents)]
    q_sigma = [torch.from_numpy(mat['q_sigma'][(0,i)]).permute(2,0,1) for i in range(nLatents)]

    params0 = {"qMu0": None, "qSVec0": q_sqrt, "qSDiag0": q_diag}
    qU = SVPosteriorOnIndPoints()
    qU.setInitialParams(initialParams=params0)
    qSigma = qU.buildQSigma();

    error = torch.tensor([(qSigma[k]-q_sigma[k]).norm() for k in range(len(qSigma))]).sum()

    assert(error<tol)

if __name__=="__main__":
    test_buildQSigma()
