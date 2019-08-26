
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
from utils import getDiagIndicesIn3DArray, build3DdiagFromDiagVector

def test_getDiagIndicesIn3DArray():
    N = 3
    M = 2
    trueDiagIndices = np.array([0, 4, 8, 9, 13, 17])

    diagIndices = getDiagIndicesIn3DArray(N=N, M=M)
    assert(((trueDiagIndices-diagIndices)**2).sum()==0)

def test_build3DdiagFromDiagVector():
    N = 3
    M = 2
    v = np.arange(M*N)
    D = build3DdiagFromDiagVector(v=v, N=N, M=M)
    trueD = np.array([[[0,0,0],[0,1,0],[0,0,2]],[[3,0,0],[0,4,0],[0,0,5]]])
    assert(((trueD-D)**2).sum()==0)

if __name__=="__main__":
    test_getDiagIndicesIn3DArray()
    test_build3DdiagFromDiagVector()
