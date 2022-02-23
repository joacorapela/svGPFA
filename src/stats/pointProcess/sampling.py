
import pdb
import random
import torch

class Sampler:

    def sampleInhomogeneousPP_thinning(self, cifTimes, cifValues, T):
        """ Thining algorithm to sample from an inhomogeneous point process. Algorithm 2 from Yuanda Chen (2016). Thinning algorithms for simulating Point Prcesses.

        :param: cifFun: Intensity function of the point process.
        :type: cifFun: function

        :param: T: the returned samples of the point process :math:`\in [0, T]`
        :type: T: double

        :param: nGrid: number of points in the grid used to search for the maximum of cifFun.
        :type: nGrid: integer

        :return: (inhomogeneous, homogeneous): samples of the inhomogeneous and homogenous point process with cif function cifFun.
        :rtype: tuple containing two lists
        """
        m = 0
        t = [0]
        s = [0]
        lambdaMax = cifValues.max()
        while s[m]<T:
            u = torch.rand(1)
            w = -torch.log(u)/lambdaMax    # w~exponential(lambdaMax)
            s.append(s[m]+w)               # {sm} homogeneous Poisson process
            D = random.uniform(0, 1)
            cifIndex = (cifTimes-s[m+1]).argmin()
            approxIntensityAtNewPoissonSpike = cifValues[cifIndex]
            if D<=approxIntensityAtNewPoissonSpike/lambdaMax: # accepting with probability
                                                              # cifF(s[m+1])/lambdaMax
                t.append(s[m+1].item())                       # {tn} inhomogeneous Poisson
                                                              # process
            m += 1
        if t[-1]<=T:
            answer = {"inhomogeneous": t[1:], "homogeneous": s[1:]}
        else:
            answer = {"inhomogeneous": t[1:-1], "homogeneous": s[1:-1]}
        return answer

    def sampleInhomogeneousPP_timeRescaling(self, cifTimes, cifValues, T):
        """ Time rescaling algorithm to sample from an inhomogeneous point
        process. Chapter 2 from Uri Eden's Point Process Notes.

        :param: cifFun: cif function of the point process.
        :type: cifFun: function

        :param: T: the returned samples of the point process :math:`\in [0, T]`
        :type: T: double

        :param: dt: spike-time resolution.
        :type: dt: float

        :return: samples of the inhomogeneous point process with cif function cifFun.
        :rtype: list
        """
        s = []
        i = 0
        dt = cifTimes[1]-cifTimes[0]
        while i<(len(cifTimes)-1):
            u = torch.rand(1)
            z = -torch.log(u)   # z~exponential(1.0)
            anInt = cifValues[i]*dt
            j = i+1
            while j<len(cifTimes) and anInt<=z:
                anInt += cifValues[j]*dt
                j += 1
            if anInt>z:
                s.append(cifTimes[j-1].item())
            i = j
        return s
