
import pdb
from abc import ABC, abstractmethod
import math
import torch

import svGPFA.utils.miscUtils

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
        qSigma = svGPFA.utils.miscUtils.buildQSigmasFromSRQSigmaVecs(srQSigmaVecs=self._srQSigmaVecs)
        return qSigma

class SVPosteriorOnIndPointsCholWithGettersAndSetters(SVPosteriorOnIndPointsChol):
    def get_flattened_params(self):
        flattened_params = []
        for k in range(len(self._qMu)):
            flattened_params.extend(self._qMu[k].flatten().tolist())
        for k in range(len(self._srQSigmaVecs)):
            flattened_params.extend(self._srQSigmaVecs[k].flatten().tolist())
        return flattened_params

    def get_flattened_params_grad(self):
        flattened_params_grad = []
        for k in range(len(self._qMu)):
            flattened_params_grad.extend(self._qMu[k].grad.flatten().tolist())
        for k in range(len(self._srQSigmaVecs)):
            flattened_params_grad.extend(self._srQSigmaVecs[k].grad.flatten().tolist())
        return flattened_params_grad

    def set_params_from_flattened(self, flattened_params):
        for k in range(len(self._qMu)):
            flattened_param = flattened_params[:self._qMu[k].numel()]
            self._qMu[k] = torch.tensor(flattened_param, dtype=torch.double).reshape(self._qMu[k].shape)
            flattened_params = flattened_params[self._qMu[k].numel():]
        for k in range(len(self._srQSigmaVecs)):
            flattened_param = flattened_params[:self._srQSigmaVecs[k].numel()]
            self._srQSigmaVecs[k] = torch.tensor(flattened_param, dtype=torch.double).reshape(self._srQSigmaVecs[k].shape)
            flattened_params = flattened_params[self._srQSigmaVecs[k].numel():]

    def set_params_requires_grad(self, requires_grad):
        for k in range(len(self._qMu)):
            self._qMu[k].requires_grad = requires_grad
        for k in range(len(self._srQSigmaVecs)):
            self._srQSigmaVecs[k].requires_grad = requires_grad

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
            dd = svGPFA.utils.miscUtils.build3DdiagFromDiagVector(v=(self._qSDiag[k].flatten())**2, M=R, N=nIndKVarRnkK)
            qSigma[k] = torch.matmul(qq, torch.transpose(a=qq, dim0=1, dim1=2)) + dd
        return(qSigma)

