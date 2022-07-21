
import pdb
import math
import torch
import scipy.stats

class GaussianProcess(object):

    def __init__(self, mean, kernel):
        self._mean = mean
        self._kernel = kernel

    def __call__(self, t, epsilon=1e-5):
        return self.eval(t=t, epsilon=epsilon)

    def eval(self, t, regularization=1e-5):
        mean = self._mean(t)
        cov = self._kernel.buildKernelMatrix(t)
        cov = cov + regularization*torch.eye(cov.shape[0])
        mn = scipy.stats.multivariate_normal(mean=mean, cov=cov)
        samples = torch.from_numpy(mn.rvs())
        return samples, mean, cov

    def mean(self, t):
        return self._mean(t)

    def std(self, t):
        Kdiag = self._kernel.buildKernelMatrixDiag(t)
        std = torch.sqrt(Kdiag)
        return std

