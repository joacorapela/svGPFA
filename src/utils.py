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
    def clocked(*args):
        t0 = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter()-t0
        name = func.__name__
        arg_str = ', '.join(repr(arg) for arg in args)
        print('[%.0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        return result
    return clocked
