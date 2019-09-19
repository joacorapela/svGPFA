
import pdb
import time
import torch

def getDiagIndicesIn3DArray(N, M):
    frameDiagIndices = torch.arange(end=N)*(N+1)
    frameStartIndices = torch.arange(end=M)*N**2
    # torch way of computing an outer sum
    diagIndices = (frameDiagIndices.reshape(-1,1)+frameStartIndices).flatten()
    answer, _ = diagIndices.sort()
    return answer

def build3DdiagFromDiagVector(v, N, M):
    assert(len(v)==N*M)
    diagIndices = getDiagIndicesIn3DArray(N=N, M=M)
    D = torch.zeros(M*N*N, dtype=torch.double)
    D[diagIndices] = v
    reshapedD = D.reshape(shape = (M, N, N))
    return reshapedD

def flattenListsOfArrays(*lists):
    aListOfArrays = []
    for arraysList in lists:
        for array in arraysList:
            aListOfArrays.append(array.flatten())
    return torch.cat(aListOfArrays)

def pinv3D(K):
    Kinv = torch.zeros(K.shape, dtype=torch.double)
    for i in range(K.shape[0]):
        Kinv[i,:,:] = torch.pinverse(K[i,:,:])
    return Kinv

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
