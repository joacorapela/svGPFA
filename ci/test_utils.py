
import sys
import os
import pdb
import math
from scipy.io import loadmat
import torch
sys.path.append("../src")
import stats.svGPFA.utils

def test_getDiagIndicesIn3DArray():
    N = 3
    M = 2
    trueDiagIndices = torch.tensor([0, 4, 8, 9, 13, 17])

    diagIndices = stats.svGPFA.utils.getDiagIndicesIn3DArray(N=N, M=M)
    assert(((trueDiagIndices-diagIndices)**2).sum()==0)

def test_build3DdiagFromDiagVector():
    N = 3
    M = 2
    v = torch.arange(M*N, dtype=torch.double)
    D = stats.svGPFA.utils.build3DdiagFromDiagVector(v=v, N=N, M=M)
    trueD = torch.tensor([[[0,0,0],[0,1,0],[0,0,2]],[[3,0,0],[0,4,0],[0,0,5]]], dtype=torch.double)
    assert(((trueD-D)**2).sum()==0)

# def test_j_cholesky():
#     tol = 1e-3
# 
#     A = torch.randn((3, 4))
#     K = torch.mm(A, A.T)
#     trueY = torch.unsqueeze(torch.tensor([1.0, 2.0, 3.0]), 1)
#     b = torch.mm(K, trueY)
#     KChol = torch.cholesky(K)
#     yTorch = torch.cholesky_solve(b, KChol)
#     yJ = stats.svGPFA.utils.j_cholesky_solve(b, KChol)
#     error = ((yTorch-yJ)**2).sum()
#     assert(error<tol)
# 
if __name__=="__main__":
    test_getDiagIndicesIn3DArray()
    test_build3DdiagFromDiagVector()
    # test_j_cholesky()
