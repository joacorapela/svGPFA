
import pdb
from abc import ABC, abstractmethod
import math
import torch

import svGPFA.utils.miscUtils

class SVPosteriorOnIndPoints(ABC):

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

class SVPosteriorOnIndPointsChol(SVPosteriorOnIndPoints):

    def __init__(self):
        super(SVPosteriorOnIndPoints, self).__init__()

    def setInitialParams(self, initial_params):
        nLatents = len(initial_params["mean"])
        self._mean = [initial_params["mean"][k] for k in range(nLatents)]
        self._cholVecs = [initial_params["cholVecs"][k] for k in range(nLatents)]

    def getParams(self):
        listOfTensors = []
        listOfTensors.extend([self._mean[k] for k in range(len(self._mean))])
        listOfTensors.extend([self._cholVecs[k] for k in
                              range(len(self._cholVecs))])
        return listOfTensors

    def getMean(self):
        return self._mean

    def buildCov(self):
        qSigma = svGPFA.utils.miscUtils.buildCovsFromCholVecs(cholVecs=self._cholVecs)
        return qSigma

class SVPosteriorOnIndPointsCholWithGettersAndSetters(SVPosteriorOnIndPointsChol):
    def get_flattened_params(self):
        flattened_params = []
        for k in range(len(self._mean)):
            flattened_params.extend(self._mean[k].flatten().tolist())
        for k in range(len(self._cholVecs)):
            flattened_params.extend(self._cholVecs[k].flatten().tolist())
        return flattened_params

    def get_flattened_params_grad(self):
        flattened_params_grad = []
        for k in range(len(self._mean)):
            flattened_params_grad.extend(self._mean[k].grad.flatten().tolist())
        for k in range(len(self._cholVecs)):
            flattened_params_grad.extend(self._cholVecs[k].grad.flatten().tolist())
        return flattened_params_grad

    def set_params_from_flattened(self, flattened_params):
        for k in range(len(self._mean)):
            flattened_param = flattened_params[:self._mean[k].numel()]
            self._mean[k] = torch.tensor(flattened_param, dtype=torch.double).reshape(self._mean[k].shape)
            flattened_params = flattened_params[self._mean[k].numel():]
        for k in range(len(self._cholVecs)):
            flattened_param = flattened_params[:self._cholVecs[k].numel()]
            self._cholVecs[k] = torch.tensor(flattened_param,
                                                   dtype=torch.double).reshape(self._cholVecs[k].shape)
            flattened_params = flattened_params[self._cholVecs[k].numel():]

    def set_params_requires_grad(self, requires_grad):
        for k in range(len(self._mean)):
            self._mean[k].requires_grad = requires_grad
        for k in range(len(self._cholVecs)):
            self._cholVecs[k].requires_grad = requires_grad


class SVPosteriorOnIndPointsRank1PlusDiag(SVPosteriorOnIndPoints):

    def __init__(self):
        super(SVPosteriorOnIndPoints, self).__init__()

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

