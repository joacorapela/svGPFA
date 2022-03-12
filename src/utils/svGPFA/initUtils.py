
import pdb
import sys
import os
import torch
import myMath.utils
import stats.svGPFA.kernelsMatricesStore
import utils.svGPFA.miscUtils

def getUniformIndPointsMeans(nTrials, nLatents, nIndPointsPerLatent, min=-1, max=1):
    indPointsMeans = [[] for r in range(nTrials)]
    for r in range(nTrials):
        indPointsMeans[r] = [[] for k in range(nLatents)]
        for k in range(nLatents):
            indPointsMeans[r][k] = torch.rand(nIndPointsPerLatent[k], 1)*(max-min)+min
    return indPointsMeans

def getConstantIndPointsMeans(constantValue, nTrials, nLatents, nIndPointsPerLatent):
    indPointsMeans = [[] for r in range(nTrials)]
    for r in range(nTrials):
        indPointsMeans[r] = [[] for k in range(nLatents)]
        for k in range(nLatents):
            indPointsMeans[r][k] = constantValue*torch.ones(nIndPointsPerLatent[k], 1, dtype=torch.double)
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

def getScaledIdentityQSigma0(scale, nTrials, nIndPointsPerLatent):
    nLatent = len(nIndPointsPerLatent)
    qSigma0 = [[None] for k in range(nLatent)]

    for k in range(nLatent):
        qSigma0[k] = torch.empty((nTrials, nIndPointsPerLatent[k], nIndPointsPerLatent[k]), dtype=torch.double)
        for r in range(nTrials):
            qSigma0[k][r,:,:] = scale*torch.eye(nIndPointsPerLatent[k], dtype=torch.double)
    return qSigma0

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

def getKernelsScaledParams0(kernels, noiseSTD):
    nLatents = len(kernels)
    kernelsParams0 = [[] for r in range(nLatents)]
    for k in range(nLatents):
        trueParams = kernels[k].getScaledParams()
        kernelsParams0[k] = noiseSTD*torch.randn(len(trueParams))+trueParams
    return kernelsParams0

def getSRQSigmaVecsFromKzz(Kzz):
    Kzz_chol = []
    for aKzz in Kzz:
        Kzz_chol.append(utils.svGPFA.miscUtils.chol3D(aKzz))
    answer = getSRQSigmaVecsFromSRMatrices(srMatrices=Kzz_chol)
    return answer

def getSRQSigmaVecsFromSRMatrices(srMatrices):
    nLatents = len(srMatrices)
    nTrials = srMatrices[0].shape[0]

    srQSigmaVec = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        nIndPointsK = srMatrices[k].shape[1]
        Pk = int((nIndPointsK+1)*nIndPointsK/2)
        srQSigmaVec[k] = torch.empty((nTrials, Pk, 1), dtype=torch.double)
        for r in range(nTrials):
            cholKR = srMatrices[k][r,:,:]
            trilIndices = torch.tril_indices(nIndPointsK, nIndPointsK)
            cholKRVec = cholKR[trilIndices[0,:], trilIndices[1,:]]
            srQSigmaVec[k][r,:,0] = cholKRVec
    return srQSigmaVec

