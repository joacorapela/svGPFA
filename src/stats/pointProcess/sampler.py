
import pdb
import random
import torch

class Sampler:

    def sampleInhomogeneousPP_thinning(self, intensityTimes, intensityValues, T):
        """ Thining algorithm to sample from an inhomogeneous point process. Algorithm 2 from Yuanda Chen (2016). Thinning algorithms for simulating Point Prcesses.

        :param: intensityFun: Intensity function of the point process.
        :type: intensityFun: function

        :param: T: the returned samples of the point process :math:`\in [0, T]`
        :type: T: double

        :param: nGrid: number of points in the grid used to search for the maximum of intensityFun.
        :type: nGrid: integer

        :return: (inhomogeneous, homogeneous): samples of the inhomogeneous and homogenous point process with intensity function intensityFun.
        :rtype: tuple containing two lists
        """
        m = 0
        t = [0]
        s = [0]
        lambdaMax = intensityValues.max()
        while s[m]<T:
            u = torch.rand(1)
            w = -torch.log(u)/lambdaMax    # w~exponential(lambdaMax)
            s.append(s[m]+w)               # {sm} homogeneous Poisson process
            D = random.uniform(0, 1)
            intensityIndex = (intensityTimes-s[m+1]).argmin()
            approxIntensityAtNewPoissonSpike = intensityValues[intensityIndex]
            if D<=approxIntensityAtNewPoissonSpike/lambdaMax: # accepting with probability
                                                              # intensityF(s[m+1])/lambdaMax
                t.append(s[m+1].item())                       # {tn} inhomogeneous Poisson
                                                              # process
            m += 1
        if t[-1]<=T:
            answer = {"inhomogeneous": t[1:], "homogeneous": s[1:]}
        else:
            answer = {"inhomogeneous": t[1:-1], "homogeneous": s[1:-1]}
        pdb.set_trace()

    def sampleInhomogeneousPP_timeRescaling(self, intensityTimes, intensityValues, T):
        """ Time rescaling algorithm to sample from an inhomogeneous point
        process. Chapter 2 from Uri Eden's Point Process Notes.

        :param: intensityFun: intensity function of the point process.
        :type: intensityFun: function

        :param: T: the returned samples of the point process :math:`\in [0, T]`
        :type: T: double

        :param: dt: spike-time resolution.
        :type: dt: float

        :return: samples of the inhomogeneous point process with intensity function intensityFun.
        :rtype: list
        """
        s = [0]
        i = 1
        dt = intensityTimes[1]-intensityTimes[0]
        while i<(len(intensityTimes)-1):
            u = torch.rand(1)
            z = -torch.log(u)   # z~exponential(1.0)
            anInt = 0
            j = i+1
            while j<len(intensityTimes) and anInt<=z:
                anInt += intensityValues[j]*dt
                j += 1
            if anInt>z:
                s.append(intensityTimes[j-1].item())
            i = j
        answer = s[1:]
        pdb.set_trace()
        return answer
