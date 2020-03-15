
import pdb
import torch
from .utils import build3DdiagFromDiagVector

class SVPosteriorOnIndPoints:

    def __init__(self):
        super(SVPosteriorOnIndPoints, self).__init__()

    def setInitialParams(self, initialParams):
        nLatents = len(initialParams["qMu0"])
        self._qMu = [initialParams["qMu0"][k] for k in range(nLatents)]
        self._qSVec = [initialParams["qSVec0"][k] for k in range(nLatents)]
        self._qSDiag = [initialParams["qSDiag0"][k] for k in range(nLatents)]

    def getParams(self):
        listOfTensors = []
        listOfTensors.extend([self._qMu[k] for k in range(len(self._qMu))])
        listOfTensors.extend([self._qSVec[k] for k in range(len(self._qSVec))])
        listOfTensors.extend([self._qSDiag[k] for k in range(len(self._qSDiag))])
        return listOfTensors

    def getQMu(self):
        return self._qMu

    def buildQSigma(self):
        R = self._qSVec[0].shape[0]
        K = len(self._qSVec)
        qSigma = [[None] for k in range(K)]
        for k in range(K):
            nIndK = self._qSDiag[k].shape[1]
            # qq \in nTrials x nInd[k] x 1
            qq = self._qSVec[k].reshape(shape=(R, nIndK, 1))
            # dd \in nTrials x nInd[k] x 1
            nIndKVarRnkK = self._qSVec[k].shape[1]
            dd = build3DdiagFromDiagVector(v=(self._qSDiag[k].flatten())**2, M=R, N=nIndKVarRnkK)
            qSigma[k] = torch.matmul(qq, torch.transpose(a=qq, dim0=1, dim1=2)) + dd
        return(qSigma)

