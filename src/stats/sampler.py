
import random
import torch

class Sampler:

    def sampleInhomogeneousPP_thinning(self, intensityFun, T, dt=.03):
        '''
        Thining algorithm to sample from an inhomogeneous point process. Algorithm 2 from Yuanda Chen (2016). Thinning algorithms for simulating Point Prcesses.

        Parameters
        ----------

        intensityFun: function
                      Intensity function of the point process.

        T: float
           The returned samples of the point process will be in [0, T]

        nGrid: integer
               number of points in the grid used to search for the maximum of intensityFun.

        Returns
        -------

        inhomogeneous: list
            samples of the inhomogeneous point process with intensity function intensityFun.
        homogeneous: list
            samples of the homogeneous that was filtered to generate the inhomogeneous point process.
        '''
        n = m = 0
        t = [0]
        s = [0]
        gridEval = intensityFun(torch.linspace(0, T, round(T/dt)))
        lambdaMax = gridEval.max()
        while s[m]<T:
            u = torch.rand()
            w = -torch.log(u)/lambdaMax    # w~exponential(lambdaMax)
            s.append(s[m]+w)               # {sm} homogeneous Poisson process
            D = random.uniform(0, 1)
            if D<=intensityFun(s[m+1])/lambdaMax:   # accepting with probability
                                                    # intensityF(s[m+1])/lambdaMax
                t.append(s[m+1])                    # {tn} inhomogeneous Poisson
                                                    # process
                n += 1
            m += 1
        if t[n]<=T:
            return({"inhomogeneous": t[1:], "homogeneous": s[1:]})
        else:
            return({"inhomogeneous": t[1:-1], "homogeneous": s[1:-1]})

    def sampleInhomogeneousPP_timeRescaling(self, intensityFun, T, dt=.03):
        '''
        Time rescaling algorithm to sample from an inhomogeneous point process. Chapter 2 from Uri Eden papers/numericalMethods/uri-eden-point-process-notes.pdf

        Parameters
        ----------

        intensityFun: function
                      Intensity function of the point process.

        T: float
           The returned samples of the point process will be in [0, T]

        nGrid: integer
               number of points in the grid used to search for spike times.

        Returns
        -------

        list
            samples of the inhomogeneous point process with intensity function intensityFun.
        '''
        t = torch.linspace(0, T, round(T/dt))
        gridEval = intensityFun(t)
        s = [0]
        i = 1
        while i<(len(t)-1):
            u = torch.rand(1)
            z = -torch.log(u)/1.0    # z~exponential(1.0)
            anInt = 0
            j = i+1
            while j<len(t) and anInt<=z:
                anInt += gridEval[j]*dt
                j += 1
            if anInt>z:
                s.append(t[j-1])
            i = j
        return s[1:]
