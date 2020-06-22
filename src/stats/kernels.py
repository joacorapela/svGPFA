
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

    def getParams(self):
        return self._params

    def setParams(self, params):
        self._params = params

    @abstractmethod
    def getNamedParams(self):
        pass

class ExponentialQuadraticKernel(Kernel):

    def __init__(self, scale=1.0, lengthScale=None, dtype=torch.double, device=torch.device("cpu")):
        if scale is not None:
            self._scale = torch.tensor(scale)
            self._scaleFixed = True
        else:
            self._scaleFixed = False

        if lengthScale is not None:
            self._lengthScale = torch.tensor(lengthScale)
            self._lengthScaleFixed = True
        else:
            self._lengthScaleFixed = False

    def buildKernelMatrix(self, X1, X2=None):
        scale, lengthScale = self._getAllParams()

        if X2 is None:
            X2 = X1
        if X1.ndim==3:
            distance = (X1-X2.transpose(1, 2))**2
        else:
            distance = (X1.reshape(-1,1)-X2.reshape(1,-1))**2
        covMatrix = scale**2*torch.exp(-.5*distance/lengthScale**2)
        return covMatrix

    def buildKernelMatrixDiag(self, X):
        scale, lengthScale = self._getAllParams()
        covMatrixDiag = scale**2*torch.ones(X.shape, dtype=X.dtype, device=X.device)
        return covMatrixDiag

    def _getAllParams(self):
        if not self._scaleFixed and not self._lengthScaleFixed:
            scale = self._params[0]
            lengthScale = self._params[1]
        elif self._scaleFixed and not self._lengthScaleFixed:
            scale = self._scale
            lengthScale = self._params[0]
        elif not self._scaleFixed and self._lengthScaleFixed:
            scale = self._params[0]
            lengthScale = self._lengthScale
        else:
            raise ValueError("Scale and lengthScale cannot be both fixed")

        return scale, lengthScale

    def getNamedParams(self):
        scale, lengthScale = self._getAllParams()
        answer = {"scale": scale, "lengthScale": lengthScale}
        return answer

class PeriodicKernel(Kernel):
    def __init__(self, scale=1.0, lengthScale=None, period=None, dtype=torch.double, device=torch.device("cpu")):
        # super(PeriodicKernel, self).__init__()
        # paramIsNone = torch.tensor([scale is None, lengthScale is None, period is None])
        # self._params = torch.nn.Parameter(torch.zeros(torch.sum(paramIsNone), dtype=dtype, device=device))

        if scale is not None:
            self._scale = torch.tensor(scale)
            self._scaleFixed = True
        else:
            self._scaleFixed = False

        if lengthScale is not None:
            self._lengthScale = torch.tensor(lengthScale)
            self._lengthScaleFixed = True
        else:
            self._lengthScaleFixed = False

        if period is not None:
            self._period = torch.tensor(period)
            self._periodFixed = True
        else:
            self._periodFixed = False

    def buildKernelMatrix(self, X1, X2=None):
        scale, lengthScale, period = self._getAllParams()
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
        scale, lengthScale, period = self._getAllParams()
        covMatrixDiag = scale**2*torch.ones(X.shape, dtype=X.dtype, device=X.device)
        return covMatrixDiag

    def _getAllParams(self):
        if not self._scaleFixed and not self._lengthScaleFixed and not self._periodFixed:
            scale = self._params[0]
            lengthScale = self._params[1]
            period = self._params[2]
        elif not self._scaleFixed and not self._lengthScaleFixed and self._periodFixed:
            scale = self._params[0]
            lengthScale = self._params[1]
            period = self._period
        elif not self._scaleFixed and self._lengthScaleFixed and not self._periodFixed:
            scale = self._params[0]
            lengthScale = self._lengthScale
            period = self._params[1]
        elif not self._scaleFixed and self._lengthScaleFixed and self._periodFixed:
            scale = self._params[0]
            lengthScale = self._lengthScale
            period = self._period
        elif self._scaleFixed and not self._lengthScaleFixed and not self._periodFixed:
            scale = self._scale
            lengthScale = self._params[0]
            period = self._params[1]
        elif self._scaleFixed and not self._lengthScaleFixed and self._periodFixed:
            scale = self._scale
            lengthScale = self._params[0]
            period = self._period
        elif self._scaleFixed and self._lengthScaleFixed and not self._periodFixed:
            scale = self._scale
            lengthScale = self._lengthScale
            period = self._params[0]
        else:
            raise ValueError("Scale and lengthScale cannot be both fixed")

        return scale, lengthScale, period

    def getNamedParams(self):
        scale, lengthScale, period = self._getAllParams()
        answer = {"scale": scale, "lengthScale": lengthScale, "period": period}
        return answer

'''
class AddDiagKernel(Kernel):
    def __init__(self, kernel, epsilon=1e-5):
        self.__kernel = kernel
        self.__epsilon = epsilon

    def buildKernelMatrix(self, X1, X2=None):
        covMatrix = self.__kernel.buildKernelMatrix(X1=X1, X2=X2)
        covMatrixPlusDiag = (covMatrix +
                             self.__epsilon*torch.eye(n=covMatrix.shape[0],
                                                      dtype=X1.dtype,
                                                      device=X.device))
        return covMatrixPlusDiag

    def buildKernelMatrixDiag(self, X):
        return self.__kernel.buildKernelMatrixDiag(X=X)

    def setParams(self, params):
        self.__kernel.setParams(params)

    def getParams(self):
        params = self.__kernel.getParams()
        return params
'''

