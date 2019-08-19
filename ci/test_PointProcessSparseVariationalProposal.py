
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
# sys.path.append('..')
from core import PointProcessSparseVariationalProposal

# import matplotlib.pyplot as plt

def test_getMeanAndVarianceAtSpikeTimes():
    tol = 1e-6
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/predict_MultiOutputGP_fromSpikes.mat")

    mat = loadmat(dataFilename)
    qMu = [mat['q_mu'][(i,0)].transpose((2,0,1)) for i in range(mat['q_mu'].shape[0])]
    qSigma = [mat['q_sigma'][(0,i)].transpose((2,0,1)) for i in range(mat['q_sigma'].shape[1])]
    C = mat["C"]
    b = np.squeeze(mat["b"])
    Kzzi = [mat['Kzzi'][(i,0)].transpose((2,0,1)) for i in range(mat['Kzzi'].shape[0])]
    Kzz = [mat['Kzz'][(i,0)].transpose((2,0,1)) for i in range(mat['Kzz'].shape[0])]
    Ktz = mat['Ktz']
    Ktt = mat['Ktt']
    mu_h = [np.squeeze(mat['mu_h'][0,i]) for i in range(mat['mu_h'].shape[1])]
    var_h = [np.squeeze(mat['var_h'][0,i]) for i in range(mat['var_h'].shape[1])]
    mu_k = [np.squeeze(mat['mu_k'][0,i]) for i in range(mat['mu_k'].shape[1])]
    var_k = [np.squeeze(mat['var_k'][0,i]) for i in range(mat['var_k'].shape[1])]
    index = [mat['index'][i,0][:,0] for i in range(mat['index'].shape[0])]

    q = PointProcessSparseVariationalProposal()
    qHMu, qHVar, qKMu, qKVar = q.getMeanAndVarianceAtSpikeTimes(qMu=qMu, qSigma=qSigma, C=C, d=b, Kzzi=Kzzi, Kzz=Kzz, Ktz=Ktz, Ktt=Ktt, neuronForSpikeIndex=index)

    for i in range(len(mu_k)):
        qKMuError = math.sqrt(np.sum((mu_k[i]-qKMu[i])**2))/mu_k[i].size
        qKVarError = math.sqrt(np.sum((var_k[i]-qKVar[i])**2))/var_k[i].size
        assert(qKMuError<tol)
        assert(qKVarError<tol)

    # pdb.set_trace()

    for i in range(len(mu_h)):
        qHMuError = math.sqrt(np.sum((mu_h[i]-qHMu[i])**2))/mu_h[i].shape[0]
        qHVarError = math.sqrt(np.sum((var_h[i]-qHVar[i])**2))/var_h[i].shape[0]
        assert(qHMuError<tol)
        assert(qHVarError<tol)

        # plt.plot(var_h[i]-qHVar[i])
        # plt.show()
        # pdb.set_trace()

    pdb.set_trace()
