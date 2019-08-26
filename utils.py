import numpy as np

def getDiagIndicesIn3DArray(N, M):
    frameDiagIndices = np.arange(N)*(N+1)
    frameStartIndices = np.arange(M)*N**2
    diagIndices = np.add.outer(frameDiagIndices, frameStartIndices).flatten()
    return np.sort(diagIndices)

def build3DdiagFromDiagVector(v, N, M):
    assert(len(v)==N*M)
    diagIndices = getDiagIndicesIn3DArray(N=N, M=M)
    D = np.zeros(M*N*N)
    D[diagIndices] = v
    reshapedD = np.reshape(a=D, newshape = (M, N, N))
    return reshapedD
