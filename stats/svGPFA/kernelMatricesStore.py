
import pdb
import torch
from abc import ABC, abstractmethod
from .utils import chol3D

class KernelMatricesStore(ABC):

    @abstractmethod
    def buildKernelsMatrices(self):
        pass

    def setKernels(self, kernels):
        self._kernels = kernels

    def setInitialParams(self, initialParams):
        self._Z = initialParams["inducingPointsLocs0"]
        for k in range(len(self._kernels)):
            self._kernels[k].setParams(initialParams["kernelsParams0"][k])

    def setIndPointsLocs(self, indPointsLocs):
        self._Z = indPointsLocs

    def getIndPointsLocs(self):
        return self._Z

    def getKernels(self):
        return self._kernels

    def getKernelsParams(self):
        answer = []
        for i in range(len(self._kernels)):
            answer.append(self._kernels[i].getParams())
        return answer

class IndPointsLocsKMS(KernelMatricesStore):

    def buildKernelsMatrices(self, epsilon=1e-5):
        nLatent = len(self._kernels)
        self._Kzz = [[None] for k in range(nLatent)]
        self._KzzChol = [[None] for k in range(nLatent)]

        for k in range(nLatent):
            self._Kzz[k] = (self._kernels[k].buildKernelMatrix(X1=self._Z[k])+
                            epsilon*torch.eye(n=self._Z[k].shape[1],
                                              dtype=torch.double))
            # self._Kzz[k] = self._kernels[k].buildKernelMatrix(X1=self._Z[k])
            self._KzzChol[k] = chol3D(self._Kzz[k]) # O(n^3)

    def getKzz(self):
        return self._Kzz

    def getKzzChol(self):
        return self._KzzChol

class IndPointsLocsAndTimesKMS(KernelMatricesStore):

    def setTimes(self, times):
        self._t = times

    def getKtz(self):
        return self._Ktz

    def getKtt(self):
        return self._Ktt

class IndPointsLocsAndAllTimesKMS(IndPointsLocsAndTimesKMS):

    def buildKernelsMatrices(self):
        # t \in nTrials x nQuad x 1
        nLatent = len(self._Z)
        self._Ktz = [[None] for k in range(nLatent)]
        self._Ktt = torch.zeros(self._t.shape[0], self._t.shape[1], nLatent, 
                                dtype=torch.double)
        for k in range(nLatent):
            self._Ktz[k] = self._kernels[k].buildKernelMatrix(X1=self._t, X2=self._Z[k])
            self._Ktt[:,:,k] = self._kernels[k].buildKernelMatrixDiag(X=self._t).squeeze()

class IndPointsLocsAndAssocTimesKMS(IndPointsLocsAndTimesKMS):

    def buildKernelsMatrices(self):
        nLatent = len(self._Z)
        nTrial = self._Z[0].shape[0]
        self._Ktz = [[[None] for tr in range(nTrial)] for k in range(nLatent)]
        self._Ktt = [[[None] for tr in  range(nTrial)] for k in range(nLatent)]

        for k in range(nLatent):
            for tr in range(nTrial):
                self._Ktz[k][tr] = self._kernels[k].buildKernelMatrix(X1=self._t[tr], X2=self._Z[k][tr,:,:])
                self._Ktt[k][tr] = self._kernels[k].buildKernelMatrixDiag(X=self._t[tr])

