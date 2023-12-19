
import pdb
from abc import ABC, abstractmethod
import math
import jax.numpy as jnp

import svGPFA.utils.miscUtils

class VariationalDist(ABC):

    def __init__(self):
        pass

#     @abstractmethod
#     def setInitialParams(self, initial_params):
#         pass

#     @abstractmethod
#     def getParams(self):
#         pass

#     @abstractmethod
#     def getMean(self):
#         pass

#     @abstractmethod
#     def buildCov(self):
#         pass

class VariationalDistChol(VariationalDist):

    def __init__(self):
        super(VariationalDist, self).__init__()

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
        n_trials = self._qSVec[0].shape[0]
        n_latents = len(self._qSVec)
        qSigma = [[None] for k in range(n_latents)]
        for k in range(n_latents):
            nIndK = self._qSDiag[k].shape[1]
            # qq \in nTrials x nInd[k] x 1
            qq = self._qSVec[k].reshape(shape=(n_trials, nIndK, 1))
            # dd \in nTrials x nInd[k] x 1
            nIndKVarRnkK = self._qSVec[k].shape[1]
            dd = svGPFA.utils.miscUtils.build3DdiagFromDiagVector(
                v=(self._qSDiag[k].flatten())**2, M=n_trials, N=nIndKVarRnkK)
            qSigma[k] = jnp.matmul(qq, jnp.transpose(qq, (0, 2, 1))) + dd
        return(qSigma)

