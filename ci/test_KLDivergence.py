
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
from core import KLDivergence

def test_evalSumAcrossLatentsTrials():
    tol = 1e-5
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/klDivergence.mat")

    mat = loadmat(dataFilename)
    Kzzi = [mat['Kzzi'][(i,0)].transpose((2,0,1)) for i in range(mat['Kzzi'].shape[0])]
    qMu = [mat['q_mu'][(i,0)].transpose((2,0,1)) for i in range(mat['q_mu'].shape[0])]
    qSigma = [mat['q_sigma'][(0,i)].transpose((2,0,1)) for i in range(mat['q_sigma'].shape[1])]
    matKLDiv = mat['kldiv']

    kl = KLDivergence()
    klDiv = kl.evalSumAcrossLatentsAndTrials(Kzzi=Kzzi, qMu=qMu, qSigma=qSigma)

    klError = abs(matKLDiv-klDiv)

    assert(klError<tol)

    pdb.set_trace()

if __name__=="__main__":
    test_evalSumAcrossLatentsTrials()
