import pdb
import sys
import os
from scipy.io import loadmat
import torch
sys.path.append("../src")
from stats.svGPFA.svPosteriorOnIndPoints import SVPosteriorOnIndPoints
import utils.svGPFA.initUtils

def test_buildQSigma():
    tol = 1e-5
    dataFilename = os.path.join(os.path.dirname(__file__), "data/get_full_from_lowplusdiag.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['q_sqrt'].shape[0]
    nTrials = mat['q_sqrt'][(0,0)].shape[2]
    q_sqrt = [torch.from_numpy(mat['q_sqrt'][(k,0)]).permute(2,0,1) for k in range(nLatents)]
    q_diag = [torch.from_numpy(mat['q_diag'][(k,0)]).permute(2,0,1) for k in range(nLatents)]
    q_sigma = [torch.from_numpy(mat['q_sigma'][(0,k)]).permute(2,0,1) for k in range(nLatents)]
    qMu0 = [[] for i in range(nLatents)]

    # qSigma0[k] \in nTrials x nInd[k] x nInd[k]
    qSigma0 = utils.svGPFA.initUtils.buildQSigmaFromQSVecAndQSDiag(qSVec=q_sqrt, qSDiag=q_diag)
    qSRSigma0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        nIndPointsK = mat['q_sqrt'][(k,0)].shape[0]
        qSRSigma0[k] = torch.empty((nTrials, nIndPointsK, nIndPointsK), dtype=torch.double)
        for r in range(nTrials):
            qSRSigma0[k][r,:,:] = torch.cholesky(qSigma0[k][r,:,:])

    params0 = {"qMu0": qMu0, "qSRSigma0": qSRSigma0}
    qU = SVPosteriorOnIndPoints()
    qU.setInitialParams(initialParams=params0)
    qSigma = qU.buildQSigma();

    error = torch.tensor([(qSigma[k]-q_sigma[k]).norm() for k in range(len(qSigma))]).sum()

    assert(error<tol)

if __name__=="__main__":
    test_buildQSigma()
