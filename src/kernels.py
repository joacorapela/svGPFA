
import pdb
from abc import ABC, abstractmethod
import math
import torch

class Kernel(ABC):

    @abstractmethod
    def buildKernelMatrix(self, X1, X2=None):
        pass

    @abstractmethod
    def buildKernelMatrixDiag(self, X):
        pass

    def setParams(self, params):
        self._params = params

    def getParams(self):
        return self._params

class ExponentialQuadraticKernel(Kernel):
    def __init__(self, scale):
        self._scale = scale

    def buildKernelMatrix(self, X1, X2=None):
        scale = self._scale
        lengthScale = self._params[0]
        if X2 is None:
            X2 = X1
        if X1.ndim==3:
            distance = (X1-X2.transpose(1, 2))**2
        else:
            distance = (X1.reshape(-1,1)-X2.reshape(1,-1))**2
        covMatrix = scale**2*torch.exp(-.5*distance/lengthScale**2)
        return covMatrix

    def buildKernelMatrixDiag(self, X):
        scale = self._scale
        covMatrixDiag = self._scale**2*torch.ones(X.shape, dtype=torch.double)
        return covMatrixDiag

class PeriodicKernel(Kernel):
    def __init__(self, scale):
        self._scale = scale

    def buildKernelMatrix(self, X1, X2=None):
        scale = self._scale
        lengthScale = self._params[0]
        period = self._params[1]
        if X2 is None:
            X2 = X1
        if X1.ndim==3:
            sDistance = X1-X2.transpose(1, 2)
        else:
            sDistance = X1.reshape(-1,1)-X2.reshape(1,-1)
        rr = math.pi*sDistance/period
        covMatrix = scale**2*torch.exp(-2*torch.sin(rr)**2/lengthScale**2)
        return covMatrix

    def buildKernelMatrixDiag(self, X):
        scale = self._scale
        covMatrixDiag = self._scale**2*torch.ones(X.shape, dtype=torch.double)
        return covMatrixDiag

