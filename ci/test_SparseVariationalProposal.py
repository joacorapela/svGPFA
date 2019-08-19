
import sys
import os
import math
from scipy.io import loadmat
import numpy as np
sys.path.append('..')
from SparseVariationalProposal import SparseVariationalProposal

def test_getMeanAndVarianceAtQuadPoints():
    tol = 1e-6
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/predict_MultiOutputGP.mat")

    mat = loadmat(dataFilename)
    qMu = [mat['q_mu'][(i,0)].transpose((2,0,1)) for i in range(mat['q_mu'].shape[0])]
    qSigma = [mat['q_sigma'][(0,i)].transpose((2,0,1)) for i in range(mat['q_sigma'].shape[1])]
    C = mat["C"]
    b = mat["b"]
    Kzzi = [mat['Kzzi'][(i,0)].transpose((2,0,1)) for i in range(mat['Kzzi'].shape[0])]
    Kzz = [mat['Kzz'][(i,0)].transpose((2,0,1)) for i in range(mat['Kzz'].shape[0])]
    Ktz = [mat['Ktz'][(i,0)].transpose((2,0,1)) for i in range(mat['Ktz'].shape[0])]
    Ktt = mat['Ktt'].transpose((2,0,1))
    mu_h = mat['mu_h'].transpose((2,0,1))
    var_h = mat['var_h'].transpose((2,0,1))

    q = SparseVariationalProposal()
    qHMu, qHVar = q.getMeanAndVarianceAtQuadPoints(qMu=qMu, qSigma=qSigma, C=C, d=b, Kzzi=Kzzi, Kzz=Kzz, Ktz=Ktz, Ktt=Ktt)

    qHMuError = math.sqrt(np.sum((mu_h-qHMu)**2))
    qHVarError = math.sqrt(np.sum((var_h-qHVar)**2))

    assert(qHMuError<tol)
    assert(qHVarError<tol)
