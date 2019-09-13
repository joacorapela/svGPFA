
import pdb
from abc import ABC, abstractmethod
import math
import torch

class Kernel(ABC):

    @abstractmethod
    def buildKernelMatrix(self, X1, X2=None):
        pass

    @abstractmethod
    def buildKernelMatrixDiag(self, t):
        pass

class ExponentialQuadraticKernel(Kernel):
    def __init__(self, scale, lengthScale):
        self.__scale = 1.0
        self.__variableParams = torch.tensor([lengthScale])

    def buildKernelMatrix(self, X1, X2=None):
        scale = self.__scale
        lengthScale = self.__variableParams[0]
        if X2 is None:
            X2 = X1
        if X1.ndim==3:
            distance = (X1-X2.transpose(1, 2))**2
        else:
            distance = (X1-X2.transpose(0, 1))**2
        covMatrix = scale**2*torch.exp(-.5*distance/lengthScale**2)
        return covMatrix

    def buildKernelMatrixDiag(self, X):
        scale = self.__scale
        covMatrixDiag = self.__scale**2*torch.ones(X.shape, dtype=torch.double)
        return covMatrixDiag

class PeriodicKernel(Kernel):
    def __init__(self, scale, lengthScale, period):
        self.__scale = scale
        self.__variableParams = torch.tensor([lengthScale, period])

    def buildKernelMatrix(self, X1, X2=None):
        scale = self.__scale
        lengthScale = self.__variableParams[0]
        period = self.__variableParams[1]
        if X2 is None:
            X2 = X1
        if X1.ndim==3:
            sDistance = X1-X2.transpose(1, 2)
        else:
            sDistance = X1-X2.transpose(0, 1)
        rr = math.pi*sDistance/period
        covMatrix = scale**2*torch.exp(-2*torch.sin(rr)**2/lengthScale**2)
        return covMatrix

    def buildKernelMatrixDiag(self, X):
        scale = self.__scale
        covMatrixDiag = self.__scale**2*torch.ones(X.shape, dtype=torch.double)
        return covMatrixDiag

