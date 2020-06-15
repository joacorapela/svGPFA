
import pdb
import sys
import os
import torch
import myMath.utils

def getIndPointLocs0(nIndPointsPerLatent, trialsLengths, firstIndPoint):
    nLatents = len(nIndPointsPerLatent)
    nTrials = len(trialsLengths)

    Z0 = [None]*nLatents
    for i in range(nLatents):
        Z0[i] = torch.empty((nTrials, nIndPointsPerLatent[i], 1), dtype=torch.double)
    for i in range(nLatents):
        for j in range(nTrials):
            Z0[i][j,:,0] = torch.linspace(firstIndPoint, trialsLengths[j], nIndPointsPerLatent[i])
    return Z0

def getSVPosteriorOnIndPointsParams0(nIndPointsPerLatent, nLatents, nTrials, scale):
    qMu0 = [None]*nLatents
    qSVec0 = [None]*nLatents
    qSDiag0 = [None]*nLatents
    for i in range(nLatents):
        qMu0[i] = torch.zeros(nTrials, nIndPointsPerLatent[i], 1, dtype=torch.double)
        qSVec0[i] = scale*torch.eye(nIndPointsPerLatent[i], 1, dtype=torch.double).repeat(nTrials, 1, 1)
        qSDiag0[i] = scale*torch.ones(nIndPointsPerLatent[i], 1, dtype=torch.double).repeat(nTrials, 1, 1)
    pdb.set_trace()
    return qMu0, qSVec0, qSDiag0

def getKernelsParams0(kernels, noiseSTD):
    nTrials = len(kernels)
    nLatents = len(kernels[0])
    kernelsParams0 = [[] for r in range(nTrials)]
    for r in range(nTrials):
        kernelsParams0[r] = [[] for r in range(nLatents)]
        for k in range(nLatents):
            trueParams = kernels[r][k].getParams()
            kernelsParams0[r][k] = noiseSTD*torch.randn(len(trueParams))+trueParams
    return kernelsParams0

