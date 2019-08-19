
import sys
import os
import math
from scipy.io import loadmat
import numpy as np
sys.path.append('..')
from core import PointProcessExpectedLogLikelihood

def test_getMeanAndVarianceAtQuadPoints():
    tol = 1e-6
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/expectedLogLik_PointProcess.mat")

    mat = loadmat(dataFilename)
    hermQuadPoints = mat["xxHerm"]
    hermQuadWeights = mat["wwHerm"]
    legQuadPoints = mat["ttQuad"]
    legQuadWeights = mat["wwQuad"]
    linkFunction = np.exp
    ell = PointProcessExpectedLogLikelihood(legQuadPoints=legQuadPoints,
                                             legQuadWeights=legQuadWeights, 
                                             hermQuadPoints=hermQuadPoints, 
                                             hermQuadWeights=hermQuadWeights, 
                                             linkFunction=linkFunction)
    res = ell.evalSumAcrossTrialsAndNeuronsWithGradientOnQ(qMu=qMu, qSigma, C, d, Kzzi, 
                                                      Kzz, KtzAtQuad, KttAtQuad,
                                                      KtzAtSpike, KttAtSpike):
