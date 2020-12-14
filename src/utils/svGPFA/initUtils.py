
import pdb
import sys
import os
import torch
import myMath.utils
import stats.svGPFA.kernelsMatricesStore

def getUniformIndPointsMeans(nTrials, nLatents, nIndPointsPerLatent, min=-1, max=1):
    indPointsMeans = [[] for r in range(nTrials)]
    for r in range(nTrials):
        indPointsMeans[r] = [[] for k in range(nLatents)]
        for k in range(nLatents):
            indPointsMeans[r][k] = torch.rand(nIndPointsPerLatent[k], 1)*(max-min)+min
    return indPointsMeans

def getZeroIndPointsMeans(nTrials, nLatents, nIndPointsPerLatent):
    indPointsMeans = [[] for r in range(nTrials)]
    for r in range(nTrials):
        indPointsMeans[r] = [[] for k in range(nLatents)]
        for k in range(nLatents):
            indPointsMeans[r][k] = torch.zeros(nIndPointsPerLatent[k], 1)
    return indPointsMeans

def getKzzChol0(kernels, kernelsParams0, indPointsLocs0, epsilon):
    indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS()
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setKernelsParams(kernelsParams=kernelsParams0)
    indPointsLocsKMS.setIndPointsLocs(indPointsLocs=indPointsLocs0)
    indPointsLocsKMS.setEpsilon(epsilon=epsilon)
    indPointsLocsKMS.buildKernelsMatrices()
    KzzChol0 = indPointsLocsKMS.getKzzChol()
    return KzzChol0

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

def buildQSigmaFromQSVecAndQSDiag(qSVec, qSDiag):
    nLatents = len(qSVec)
    nTrials = qSVec[0].shape[0]
    qSigma = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        nIndK = qSDiag[k].shape[1]
        # qq \in nTrials x nInd[k] x 1
        qq = qSVec[k].reshape(shape=(nTrials, nIndK, 1))
        # dd \in nTrials x nInd[k] x 1
        nIndKVarRnkK = qSVec[k].shape[1]
        dd = build3DdiagFromDiagVector(v=(qSDiag[k].flatten())**2, M=nTrials, N=nIndKVarRnkK)
        # qSigma[k] \in nTrials x nInd[k] x nInd[k]
        qSigma[k] = torch.matmul(qq, torch.transpose(a=qq, dim0=1, dim1=2)) + dd
    return(qSigma)

def getSRQSigmaVecsFromSRMatrices(srMatrices):
    nLatents = len(srMatrices)
    nTrials = srMatrices[0].shape[0]

    srQSigmaVec = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        nIndPointsK = srMatrices[k].shape[1]
        Pk = int((nIndPointsK+1)*nIndPointsK/2)
        srQSigmaVec[k] = torch.empty((nTrials, Pk), dtype=torch.double)
        for r in range(nTrials):
            cholKR = srMatrices[k][r,:,:]
            trilIndices = torch.tril_indices(nIndPointsK, nIndPointsK)
            cholKRVec = cholKR[trilIndices[0,:], trilIndices[1,:]]
            srQSigmaVec[k][r,:] = cholKRVec
    return srQSigmaVec

def getSRQSigmaVec(qSVec, qSDiag):
    nLatents = len(qSVec)
    nTrials = qSVec[0].shape[0]
    qSigma = buildQSigmaFromQSVecAndQSDiag(qSVec=qSVec, qSDiag=qSDiag)
    srQSigmaVec = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        nIndPointsK = qSigma[k].shape[1]
        Pk = int((nIndPointsK+1)*nIndPointsK/2)
        srQSigmaVec[k] = torch.empty((nTrials, Pk), dtype=torch.double)
        for r in range(nTrials):
            cholKR = torch.cholesky(qSigma[k][r,:,:])
            trilIndices = torch.tril_indices(nIndPointsK, nIndPointsK)
            cholKRVec = cholKR[trilIndices[0,:], trilIndices[1,:]]
            srQSigmaVec[k][r,:] = cholKRVec
    return srQSigmaVec

# def getIndPointLocs0(nIndPointsPerLatent, trialsLengths, firstIndPointLoc):
#     nLatents = len(nIndPointsPerLatent)
#     nTrials = len(trialsLengths)
#
#     Z0 = [[] for k in range(nLatents)]
#     for k in range(nLatents):
#         Z0[k] = torch.empty((nTrials, nIndPointsPerLatent[k], 1), dtype=torch.double)
#         for r in range(nTrials):
#             Z0[k][r,:,0] = torch.linspace(firstIndPointLoc, trialsLengths[r], nIndPointsPerLatent[k])
#     return Z0

def getSVPosteriorOnIndPointsParams0(nIndPointsPerLatent, nLatents, nTrials, scale):
    qMu0 = [[] for k in range(nLatents)]
    qSVec0 = [[] for k in range(nLatents)]
    qSDiag0 = [[] for k in range(nLatents)]
    for k in range(nLatents):
        # qMu0[k] = torch.rand(nTrials, nIndPointsPerLatent[k], 1, dtype=torch.double)
        qMu0[k] = torch.zeros(nTrials, nIndPointsPerLatent[k], 1, dtype=torch.double)
        qSVec0[k] = scale*torch.eye(nIndPointsPerLatent[k], 1, dtype=torch.double).repeat(nTrials, 1, 1)
        qSDiag0[k] = scale*torch.ones(nIndPointsPerLatent[k], 1, dtype=torch.double).repeat(nTrials, 1, 1)
    return qMu0, qSVec0, qSDiag0

def getKernelsParams0(kernels, noiseSTD):
    nLatents = len(kernels)
    kernelsParams0 = [[] for r in range(nLatents)]
    for k in range(nLatents):
        trueParams = kernels[k].getParams()
        kernelsParams0[k] = noiseSTD*torch.randn(len(trueParams))+trueParams
    return kernelsParams0

