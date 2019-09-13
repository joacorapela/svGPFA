
import sys
import os
import pdb
import math
from scipy.io import loadmat
import torch
from approxPosteriorForH import PointProcessApproxPosteriorForH
from inducingPointsPrior import InducingPointsPrior
from kernelMatricesStore import PointProcessKernelMatricesStore

# import matplotlib.pyplot as plt

def test_getMeanAndVarianceAtSpikeTimes():
    tol = 1e-6
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatent = mat['spikeKtz'].shape[0]
    nTrials = mat['spikeKtz'].shape[1]
    qMu = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSVec = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    qSDiag = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    Kzzi = [torch.from_numpy(mat['Kzzi'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    Kzz = [torch.from_numpy(mat['Kzz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    Ktz = [[torch.from_numpy(mat['spikeKtz'][k,tr]).type(torch.DoubleTensor) for tr in range(nTrials)] for k in range(nLatent)]
    Ktt = [[torch.from_numpy(mat['spikeKtt'][k,tr]).type(torch.DoubleTensor) for tr in range(nTrials)] for k in range(nLatent)]
    mu_h = [torch.from_numpy(mat['mu_h_Spikes'][0,i]).type(torch.DoubleTensor).squeeze() for i in range(nTrials)]
    var_h = [torch.from_numpy(mat['var_h_Spikes'][0,i]).type(torch.DoubleTensor).squeeze() for i in range(nTrials)]
    # mu_k = [torch.from_numpy(mat['mu_k'][0,i]).type(torch.DoubleTensor).squeeze() for i in range(mat['mu_k'].shape[1])]
    # var_k = [torch.from_numpy(mat['var_k'][0,i]).type(torch.DoubleTensor).squeeze() for i in range(mat['var_k'].shape[1])]
    index = [torch.from_numpy(mat['index'][i,0][:,0]).type(torch.ByteTensor) for i in range(nTrials)]

    qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
    kernelMatricesStore = PointProcessKernelMatricesStore(Kzz=Kzz, Kzzi=Kzzi, quadKtz=None, quadKtt=None, spikeKtz=Ktz, spikeKtt=Ktt)
    qH = PointProcessApproxPosteriorForH(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore, neuronForSpikeIndex=index)

    # qHMu, qHVar, qKMu, qKVar = q.getMeanAndVarianceAtSpikeTimes(qMu=qMu, qSigma=qSigma, C=C, d=b, Kzzi=Kzzi, Kzz=Kzz, Ktz=Ktz, Ktt=Ktt, neuronForSpikeIndex=index)
    qHMu, qHVar = qH.getMeanAndVarianceAtSpikeTimes()

    # for i in range(len(mu_k)):
        # qKMuError = math.sqrt(np.sum((mu_k[i]-qKMu[i])**2))/mu_k[i].size
        # qKVarError = math.sqrt(np.sum((var_k[i]-qKVar[i])**2))/var_k[i].size
        # assert(qKMuError<tol)
        # assert(qKVarError<tol)

    # pdb.set_trace()

    for i in range(len(mu_h)):
        qHMuError = math.sqrt(torch.sum((mu_h[i]-qHMu[i])**2))/mu_h[i].shape[0]
        qHVarError = math.sqrt(torch.sum((var_h[i]-qHVar[i])**2))/var_h[i].shape[0]
        assert(qHMuError<tol)
        assert(qHVarError<tol)

        # plt.plot(var_h[i]-qHVar[i])
        # plt.show()
        # pdb.set_trace()

if __name__=="__main__":
    test_getMeanAndVarianceAtSpikeTimes()
