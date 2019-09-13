
import pdb
from abc import ABC, abstractmethod
import torch
from utils import pinv3D

class KernelMatricesStore(ABC):

    def __init__(self, kernels, Z):
        # Z[k] \in nTrials x nInd[k] x 1
        self._kernels = kernels
        self._Z = Z
        self.__buildZKernelMatrices()

    def buildKernelMatrices(self, t):
        self.__buildZKernelMatrices()
        self.__buildZTKernelMatrices(t=t)

    def getZ(self):
        return self._Z

    def getKzz(self):
        return self._Kzz

    def getKzzi(self):
        return self._Kzzi

    def getKtz(self):
        return self._Ktz

    def getKtt(self):
        return self._Ktt

    def __buildZKernelMatrices(self, epsilon=1e-5):
        nLatent = len(self._kernels)
        self._Kzz = [[None] for k in range(nLatent)]
        self._Kzzi = [[None] for k in range(nLatent)]

        for k in range(nLatent):
            self._Kzz[k] = self._kernels[k].buildKernelMatrix(X1=self._Z[k])+epsilon*torch.eye(n=self._Z[k].shape[1], dtype=torch.double)
            self._Kzzi[k] = pinv3D(self._Kzz[k])

class KernelMatricesStoreAllNeuronsAllTimes(KernelMatricesStore):
    def __init__(self, kernels, Z, t):
        super().__init__(kernels=kernels, Z=Z)
        self.__buildZTKernelMatrices(t=t)

    def __buildZTKernelMatrices(self, t):
        # t \in nTrials x nQuad x 1
        nLatent = len(self._Z)
        self._Ktz = [[None] for k in range(nLatent)]
        self._Ktt = torch.zeros(t.shape[0], t.shape[1], nLatent, dtype=torch.double)

        for k in range(nLatent):
            self._Ktz[k] = self._kernels[k].buildKernelMatrix(X1=t, X2=self._Z[k])
            self._Ktt[:,:,k] = self._kernels[k].buildKernelMatrixDiag(X=t).squeeze()

class KernelMatricesStoreAllNeuronsAssociatedTimes(KernelMatricesStore):
    def __init__(self, kernels, Z, Y):
        super().__init__(kernels=kernels, Z=Z)
        self.__buildZTKernelMatrices(Y=Y)

    def __buildZTKernelMatrices(self, Y):
        # Y[tr] \in nSpikes of all neurons in trial tr
        nLatent = len(self._Z)
        nTrial = self._Z[0].shape[0]
        self._Ktz = [[[None] for tr in range(nTrial)] for k in range(nLatent)]
        self._Ktt = [[[None] for tr in  range(nTrial)] for k in range(nLatent)]

        for k in range(nLatent):
            for tr in range(nTrial):
                self._Ktz[k][tr] = self._kernels[k].buildKernelMatrix(X1=Y[tr], X2=self._Z[k][tr,:,:])
                self._Ktt[k][tr] = self._kernels[k].buildKernelMatrixDiag(X=Y[tr])
