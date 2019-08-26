
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
from core import PointProcessExpectedLogLikelihood

def test_evalSumAcrossTrialsAndNeuronsWithGradientOnQ():
    tol = 1e-6
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/expectedLogLik_PointProcess.mat")

    mat = loadmat(dataFilename)
    hermQuadPoints = mat["xxHerm"]
    hermQuadWeights = mat["wwHerm"]
    legQuadPoints = np.transpose(mat["ttQuad"], (2, 0, 1))
    legQuadWeights = np.transpose(mat["wwQuad"], (2, 0, 1))
    linkFunction = np.exp
    qHMeanAtQuad = np.transpose(mat["mu_h_Quad"], [2, 0, 1])
    qHVarAtQuad = np.transpose(mat["var_h_Quad"], [2, 0, 1])
    qHMeanAtSpike = mat["mu_h_Spikes"]
    qHVarAtSpike = mat["var_h_Spikes"]
    lik_pp=mat["lik_pp"]
    ell = PointProcessExpectedLogLikelihood(legQuadPoints=legQuadPoints,
                                             legQuadWeights=legQuadWeights, 
                                             hermQuadPoints=hermQuadPoints, 
                                             hermQuadWeights=hermQuadWeights, 
                                             linkFunction=linkFunction)
    sELL = ell.evalSumAcrossTrialsAndNeuronsWithGradientOnQ(
            qHMeanAtQuad, 
            qHVarAtQuad, 
            qHMeanAtSpike, 
            qHVarAtSpike)

    sELLerror = abs(sELL-lik_pp)

    assert(sELLerror<tol)

    pdb.set_trace()

if __name__=="__main__":
    test_evalSumAcrossTrialsAndNeuronsWithGradientOnQ()
