
import sys
import os
import pdb
import math
from scipy.io import loadmat
import torch
from utils import getDiagIndicesIn3DArray, build3DdiagFromDiagVector

def test_getDiagIndicesIn3DArray():
    N = 3
    M = 2
    trueDiagIndices = torch.tensor([0, 4, 8, 9, 13, 17])

    diagIndices = getDiagIndicesIn3DArray(N=N, M=M)
    assert(((trueDiagIndices-diagIndices)**2).sum()==0)

def test_build3DdiagFromDiagVector():
    N = 3
    M = 2
    v = torch.arange(M*N, dtype=torch.double)
    D = build3DdiagFromDiagVector(v=v, N=N, M=M)
    trueD = torch.tensor([[[0,0,0],[0,1,0],[0,0,2]],[[3,0,0],[0,4,0],[0,0,5]]], dtype=torch.double)
    assert(((trueD-D)**2).sum()==0)

if __name__=="__main__":
    test_getDiagIndicesIn3DArray()
    test_build3DdiagFromDiagVector()
