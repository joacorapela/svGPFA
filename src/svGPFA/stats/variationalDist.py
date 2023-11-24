
import pdb
from abc import ABC, abstractmethod
import math
import torch

import svGPFA.utils.miscUtils

class VariationalDist(ABC):

    @abstractmethod
    def setInitialParams(self, initial_params):
        pass

    @abstractmethod
    def getParams(self):
        pass

    @abstractmethod
    def getMean(self):
        pass

    @abstractmethod
    def buildCov(self):
        pass

class VariationalDistChol(VariationalDist):

    def __init__(self):
        super(VariationalDist, self).__init__()

    def setInitialParams(self, initial_params):
        nLatents = len(initial_params["mean"])
        self._mean = [initial_params["mean"][k] for k in range(nLatents)]
        self._cholVecs = [initial_params["cholVecs"][k] for k in range(nLatents)]
        self.buildCov()

    def getParams(self):
        listOfTensors = []
        listOfTensors.extend([self._mean[k] for k in range(len(self._mean))])
        listOfTensors.extend([self._cholVecs[k] for k in
                              range(len(self._cholVecs))])
        return listOfTensors

    def getMean(self):
        return self._mean

    def getCov(self):
        return self._cov

    def buildCov(self):
        self._cov = svGPFA.utils.miscUtils.buildCovsFromCholVecs(cholVecs=self._cholVecs)

class VariationalDistRank1PlusDiag(VariationalDist):

    def __init__(self):
        super(VariationalDist, self).__init__()

    def setInitialParams(self, initial_params):
        self._mean = initial_params["mean"]
        self._qSVec = initial_params["qSVec0"]
        self._qSDiag = initial_params["qSDiag0"]

    def getParams(self):
        listOfTensors = []
        listOfTensors.extend([self._mean[k] for k in range(len(self._mean))])
        listOfTensors.extend([self._qSVec[k] for k in range(len(self._qSVec))])
        listOfTensors.extend([self._qSDiag[k]
                              for k in range(len(self._qSDiag))])
        return listOfTensors

    def getMean(self):
        return self._mean

    def buildCov(self):
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

