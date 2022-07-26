
import sys
import os
import pdb
import math
from scipy.io import loadmat
import torch
import svGPFA.utils.miscUtils

def test_getPropSamplesCovered():
    N = 100
    tol = .1

    mean = torch.rand(size=(N,))*2-1
    std = torch.rand(size=(N,))*0.3
    sample = torch.normal(mean=mean, std=std)
    propSamplesCovered = svGPFA.utils.miscUtils.getPropSamplesCovered(sample=sample, mean=mean, std=std, percent=.95)
    assert(.95-tol<propSamplesCovered and propSamplesCovered<tol+.95)

def test_getDiagIndicesIn3DArray():
    N = 3
    M = 2
    trueDiagIndices = torch.tensor([0, 4, 8, 9, 13, 17])

    diagIndices = svGPFA.utils.miscUtils.getDiagIndicesIn3DArray(N=N, M=M)
    assert(((trueDiagIndices-diagIndices)**2).sum()==0)

def test_build3DdiagFromDiagVector():
    N = 3
    M = 2
    v = torch.arange(M*N, dtype=torch.double)
    D = svGPFA.utils.miscUtils.build3DdiagFromDiagVector(v=v, N=N, M=M)
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
    test_getPropSamplesCovered()
