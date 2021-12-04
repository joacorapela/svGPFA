
import pdb
from abc import ABC, abstractmethod
import math
import torch

import utils.svGPFA.miscUtils

class SVPosteriorOnIndPoints(ABC):

    @abstractmethod
    def setInitialParams(self, initialParams):
        pass

    @abstractmethod
    def getParams(self):
        pass

    @abstractmethod
    def getQMu(self):
        pass

    @abstractmethod
    def buildQSigma(self):
        pass

class SVPosteriorOnIndPointsChol(SVPosteriorOnIndPoints):

    def __init__(self):
        super(SVPosteriorOnIndPoints, self).__init__()

    def setInitialParams(self, initialParams):
        nLatents = len(initialParams["qMu0"])
        self._qMu = [initialParams["qMu0"][k] for k in range(nLatents)]
        self._srQSigmaVecs = [initialParams["srQSigma0Vecs"][k] for k in range(nLatents)]

    def getParams(self):
        listOfTensors = []
        listOfTensors.extend([self._qMu[k] for k in range(len(self._qMu))])
        listOfTensors.extend([self._srQSigmaVecs[k] for k in range(len(self._srQSigmaVecs))])
        return listOfTensors

    def getQMu(self):
        return self._qMu

    def buildQSigma(self):
        qSigma = utils.svGPFA.miscUtils.buildQSigmasFromSRQSigmaVecs(srQSigmaVecs=self._srQSigmaVecs)
        return qSigma


class SVPosteriorOnIndPointsRank1PlusDiag(SVPosteriorOnIndPoints):

    def __init__(self):
        super(SVPosteriorOnIndPoints, self).__init__()

    def setInitialParams(self, initialParams):
        self._qMu = initialParams["qMu0"]
        self._qSVec = initialParams["qSVec0"]
        self._qSDiag = initialParams["qSDiag0"]

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
            dd = utils.svGPFA.miscUtils.build3DdiagFromDiagVector(v=(self._qSDiag[k].flatten())**2, M=R, N=nIndKVarRnkK)
            qSigma[k] = torch.matmul(qq, torch.transpose(a=qq, dim0=1, dim1=2)) + dd
        return(qSigma)

