
import pdb
from abc import ABCMeta, abstractmethod
import math
import numpy as  np

class Kernel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def _k(self, x1, x2, params):
        return

    @abstractmethod
    def _kGrad(self, x1, x2, params, i):
        return

    def buildKSample(self, x1, x2, params):
        K = np.empty((len(x1), len(x2)))
        K[:] = np.nan
        for i in range(len(x1)):
            for j in range(len(x2)):
                value = self._k(x1=x1[i], x2=x2[j], params=params)
                K[i,j]=value
        return K

    def buildKSampleGrad(self, x1, x2, params):
        answer = []
        for l in range(len(params)):
            emptyMatrix = np.empty((len(x1), len(x2)))
            emptyMatrix[:] = np.nan
            answer.append(emptyMatrix)
        for i in range(len(x1)):
            for j in range(len(x2)):
                grad = self._kGrad(x1=x1[i], x2=x2[j], params=params)
                for l in range(len(grad)):
                    answer[l][i,j]=grad[l]
        return answer

class SquaredExponentialKernel(Kernel):

    def _k(self, x1, x2, params):
        l = params[0]
        s2f = params[1]**2
        s2n = params[2]**2
        answer = s2f*math.exp(-1/(2*l**2)*(x2-x1)**2)

        if x1==x2:
            answer = answer + s2n
        return answer

    def _kGrad(self, x1, x2, params):
        l = params[0]
        sf = params[1]
        sn = params[2]

        dl = sf**2/l**3*(x1-x2)**2*math.exp(-1/(2*l**2)*(x1-x2)**2)
        dsf = 2*math.exp(-1/(2*l**2)*(x1-x2)**2)*sf
        dsn = 2*(1 if x1==x2 else 0)*sn
        answer = np.array([dl, dsf, dsn])

        return(answer)

class PeriodicRandomFunctionKernel(Kernel):

    def _k(self, x1, x2, params):
        l = params[0]
        s2f = params[1]**2
        s2n = params[2]**2
        answer = s2f*math.exp(-(2*math.sin((x2-x1)/2)**2)/l**2)

        if x1==x2:
            answer = answer + s2n
        return answer

    def _kGrad(self, x1, x2, params):
        l = params[0]
        sf = params[1]
        sn = params[2]

        dl = sf**2*math.exp(-(2*math.sin((x2-x1)/2)**2)/l**2)*4*math.sin((x2-x1)/2)**2/l**3
        dsf = 2*sf*math.exp(-(2*math.sin((x2-x1)/2)**2)/l**2)
        dsn = 2*(1 if x1==x2 else 0)*sn
        answer = np.array([dl, dsf, dsn])

        return(answer)
