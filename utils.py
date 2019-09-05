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

def flattenListsOfArrays(self, *lists):
    aListOfArrays = []
    for arraysList in lists:
        for array in arraysList:
            aListOfArrays.append(array.flatten())
    return torch.cat(aListOfArrays)

