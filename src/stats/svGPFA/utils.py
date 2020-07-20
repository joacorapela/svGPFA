
import pdb
import time
import torch
import numpy as np

'''
def j_cholesky_solve(b, u, upper=False):
    # solves (uu^T) * c = b
    # first compute the forward substitution   u   * y = b
    # second compute the backward substitution u^T * c = y
    if upper==True:
        raise NotIpmplemented("option upper=True has not been implemented yet")
    if b.ndim==2 and u.ndim=2:
        return j_cholesky_solve2D(b=b, u=u)
    elif b.dim=3 and u.ndim=3:

    else:
        raise ValueError("Incorrect number of dimensions in b and/or u")

    y = forwardSubstitution(b=b, u=u)
    c = backSubstitution(b=y, u=u.T)
    return c

def forwardSubstitution(b, u):
    # solves u * y = b where u is a lower triangular matrix
    # u \in n x n
    # y \in n x k
    # b \in n x k
    n = u.shape[0]
    k = b.shape[1]
    y = torch.zeros((n, k), device=b.device)
    for j in range(k):
        y[0,j] = b[0,j]/u[0,0]
        for i in range(n):
            y[i,j] = b[i,j]
            for l in range(i):
                y[i,j] -= u[i,l]*y[l,j]
            y[i,j] /= u[i,i]
    return y

def backSubstitution(b, u):
    # solves u * y = b where u is an upper triangular matrix
    # u \in n x n
    # y \in n x k
    # b \in n x k
    n = u.shape[0]
    k = b.shape[1]
    y = torch.zeros((n, k), device=b.device)
    for j in range(k):
        y[n-1,j] = b[n-1,j]/u[n-1,n-1]
        for i in range(n-2, -1, -1):
            y[i,j] = b[i,j]
            for l in range(i+1, n):
                y[i,j] -= u[i,l]*y[l,j]
            y[i,j] /= u[i,i]
    return y
'''

def getDiagIndicesIn3DArray(N, M, device=torch.device("cpu")):
    frameDiagIndices = torch.arange(end=N, device=device)*(N+1)
    frameStartIndices = torch.arange(end=M, device=device)*N**2
    # torch way of computing an outer sum
    diagIndices = (frameDiagIndices.reshape(-1,1)+frameStartIndices).flatten()
    answer, _ = diagIndices.sort()
    return answer

def build3DdiagFromDiagVector(v, N, M):
    assert(len(v)==N*M)
    diagIndices = getDiagIndicesIn3DArray(N=N, M=M)
    D = torch.zeros(M*N*N, dtype=v.dtype, device=v.device)
    D[diagIndices] = v
    reshapedD = D.reshape(shape = (M, N, N))
    return reshapedD

def flattenListsOfArrays(*lists):
    aListOfArrays = []
    for arraysList in lists:
        for array in arraysList:
            aListOfArrays.append(array.flatten())
    return torch.cat(aListOfArrays)

# def chol3D(K):
#     Kchol = torch.zeros(K.shape, dtype=K.dtype, device=K.device)
#     for i in range(K.shape[0]):
#         Kchol[i,:,:] = torch.cholesky(K[i,:,:])
#     return Kchol

def pinv3D(K):
    Kpinv = torch.zeros(K.shape, dtype=K.dtype, device=K.device)
    for i in range(K.shape[0]):
        tol = np.max(np.array(K.shape))*np.spacing(torch.norm(K).item())
        Kpinv[i,:,:] = torch.pinverse(K[i,:,:], rcond=tol)
    return Kpinv

def clock(func):
    def clocked(*args,**kargs):
        t0 = time.perf_counter()
        result = func(*args,**kargs)
        elapsed = time.perf_counter()-t0
        name = func.__name__
        if len(args)>0:
            arg_str = ', '.join(repr(arg) for arg in args)
        else:
            arg_str = None
        if len(kargs)>0:
            keys = kargs.keys()
            values = kargs.values()
            karg_str = ', '.join(key + "=" + repr(value) for key in keys for value in values)
        else:
            karg_str = None
        if arg_str is not None and karg_str is not None:
            print('[%0.8fs] %s(%s,%s) -> %r' % (elapsed, name, arg_str, karg_str, result))
        elif arg_str is not None:
            print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        elif karg_str is not None:
            print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, karg_str, result))
        else:
            print('[%0.8fs] %s() -> %r' % (elapsed, name, result))
        return result
    return clocked
