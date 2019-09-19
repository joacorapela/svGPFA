
import sys
import os
from scipy.io import loadmat
import torch
from inducingPointsPrior import InducingPointsPrior

def test_buildQSigma():
    tol = 1e-5
    dataFilename = os.path.expanduser('data/get_full_from_lowplusdiag.mat')

    mat = loadmat(dataFilename)
    nLatents = mat['q_sqrt'].shape[1]
    q_sqrt = [torch.from_numpy(mat['q_sqrt'][(0,i)]).permute(2,0,1) for i in range(nLatents)]
    q_diag = [torch.from_numpy(mat['q_diag'][(0,i)]).permute(2,0,1) for i in range(nLatents)]
    q_sigma = [torch.from_numpy(mat['q_sigma'][(0,i)]).permute(2,0,1) for i in range(nLatents)]

    varRnk = torch.ones(3,dtype=torch.uint8)
    qU = InducingPointsPrior(qMu=None, qSVec=q_sqrt, qSDiag=q_diag, varRnk=varRnk)
    qSigma = qU.buildQSigma();

    error = torch.tensor([(qSigma[k]-q_sigma[k]).norm() for k in range(len(qSigma))]).sum()

    assert(error<tol)

if __name__=="__main__":
    main(sys.argv)
