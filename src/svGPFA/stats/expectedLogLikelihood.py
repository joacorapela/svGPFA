
import pdb
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
# import warnings

class ExpectedLogLikelihood(ABC):

    def __init__(self, preIntensityQuadTimes, linkFunction):
        self._preIntensityQuadTimes = preIntensityQuadTimes
        self._linkFunction = linkFunction

    @abstractmethod
    def evalSumAcrossTrialsAndNeurons(self, posteriorOnLatentsStats=None):
        pass


    @abstractmethod
    def computePosteriorOnLatentsStats(self):
        pass

    @abstractmethod
    def setMeasurements(self, measurements):
        pass

    @abstractmethod
    def setIndPointsLocs(self, locs):
        pass

    @abstractmethod
    def setKernels(self, kernels):
        pass

    @abstractmethod
    def setInitialParams(self, initial_params):
        pass

    @abstractmethod
    def setELLCalculationParams(self, eLLCalculationParams):
        pass

    def sampleIFs(self, times, nudget=1e-3):
        h = self._preIntensityQuadTimes.sample(times=times, nudget=nudget)
        nTrials = len(h)
        answer = [self._linkFunction(h[r]) for r in range(nTrials)]
        return answer

    def computeExpectedPosteriorIFs(self, times):
        # h \in nTrials x times x nNeurons
        eMean, eVar = self._preIntensityQuadTimes.predict(times=times)
        nTrials = eMean.shape[0]
        nNeurons = eMean.shape[2]
        answer = [[self._linkFunction(eMean[r, :, n]+0.5*eVar[r, :, n])
                   for n in range(nNeurons)] for r in range(nTrials)]
        return answer

    def getVariationalDistParams(self):
        return self._preIntensityQuadTimes.getVariationalDistParams()

    def getPreIntensityParams(self):
        return self._preIntensityQuadTimes.getParams()

    def computeEmbeddingsMeansAndVarsAtTimes(self, times):
        return self._preIntensityQuadTimes.computeMeansAndVarsAtTimes(times)

    def getIndPointsLocs(self):
        return self._preIntensityQuadTimes.getIndPointsLocs()

    def getKernels(self):
        return self._preIntensityQuadTimes.getKernels()

    def getKernelsParams(self):
        return self._preIntensityQuadTimes.getKernelsParams()

    def predictLatents(self, times):
        return self._preIntensityQuadTimes.predictLatents(times=times)

    def predictEmbedding(self, times):
        return self._preIntensityQuadTimes.predict(times=times)

    def setPriorCovRegParam(self, priorCovRegParam):
        self._preIntensityQuadTimes.setPriorCovRegParam(priorCovRegParam=priorCovRegParam)

class PointProcessELL(ExpectedLogLikelihood):
    def __init__(self, preIntensityQuadTimes, preIntensitySpikesTimes, linkFunction):
        super().__init__(preIntensityQuadTimes=preIntensityQuadTimes, linkFunction=linkFunction)
        self._preIntensitySpikesTimes = preIntensitySpikesTimes

    def evalSumAcrossTrialsAndNeurons(self, variational_mean, variational_cov,
                                      C, d, kernels_matrices):
        eMeanQuadTimes, eVarQuadTimes = \
            self._preIntensityQuadTimes.computeMeansAndVars(
                variational_mean=variational_mean,
                variational_cov=variational_cov, C=C, d=d,
                Kzz=kernels_matrices["Kzz"],
                Kzz_inv=kernels_matrices["Kzz_inv"],
                Ktz=kernels_matrices["Ktz_quad"],
                KttDiag=kernels_matrices["KttDiag_quad"])
        eMeanSpikesTimes, eVarSpikesTimes = \
            self._preIntensitySpikesTimes.computeMeansAndVars(
                variational_mean=variational_mean,
                variational_cov=variational_cov, C=C, d=d,
                Kzz=kernels_matrices["Kzz"],
                Kzz_inv=kernels_matrices["Kzz_inv"],
                Ktz=kernels_matrices["Ktz_spike"],
                KttDiag=kernels_matrices["KttDiag_spike"])
        nTrials = len(eMeanQuadTimes)
        eLinkValues = self._getELinkValues(eMean=eMeanQuadTimes,
                                           eVar=eVarQuadTimes)
        eLogLinkValues = self._getELogLinkValues(eMean=eMeanSpikesTimes,
                                                 eVar=eVarSpikesTimes)
        # self._legQuadWeights[r] \in nQuadHerm x 1
        # eLinkValues[r] \in  nQuadHerm x nNeurons
        # aux1[r] \in  1 x nNeurons
        aux1 = [jnp.matmul(self._legQuadWeights[r].T, eLinkValues[r]) for r in range(nTrials)]
        # sELLTerm1 = jnp.sum(jnp.cat([aux1[r] for r in range(nTrials)]))
        sELLTerm1 = jnp.sum(jnp.concatenate([aux1[r] for r in range(nTrials)]))
        sELLTerm2 = jnp.sum(eLogLinkValues)
        answer = -sELLTerm1+sELLTerm2
        return answer

    def computePosteriorOnLatentsStats(self):
        quadTimesStats = self._preIntensityQuadTimes.\
            computePosteriorOnLatentsStats()
        spikesTimesStats = self._preIntensitySpikesTimes.\
            computePosteriorOnLatentsStats()
        answer = {"quadTimes": quadTimesStats, "spikesTimes": spikesTimesStats}
        return answer

    def setMeasurements(self, measurements):

        stackedSpikeTimes, neuronForSpikeIndex = \
            self.__stackSpikeTimes(spikeTimes=measurements)
        # self._preIntensitySpikesTimes.setTimes(times=stackedSpikeTimes)
        self._preIntensitySpikesTimes.setNeuronForSpikeIndex(neuronForSpikeIndex=
                                                            neuronForSpikeIndex)

    def __stackSpikeTimes(self, spikeTimes):
        # spikeTimes list[nTrials][nNeurons][nSpikes]
        nTrials = len(spikeTimes)
        stackedSpikeTimes = [[] for i in range(nTrials)]
        neuronForSpikeIndex = [[] for i in range(nTrials)]
        for trialIndex in range(nTrials):
            aList = [spikeTime
                     for neuronIndex in range(len(spikeTimes[trialIndex]))
                     for spikeTime in spikeTimes[trialIndex][neuronIndex]]
            stackedSpikeTimes[trialIndex] = jnp.array(aList)
            aList = [neuronIndex
                     for neuronIndex in range(len(spikeTimes[trialIndex]))
                     for spikeTime in spikeTimes[trialIndex][neuronIndex]]
            neuronForSpikeIndex[trialIndex] = jnp.array(aList)
        return stackedSpikeTimes, neuronForSpikeIndex

    def setIndPointsLocs(self, locs):
        self._preIntensityQuadTimes.setIndPointsLocs(locs=locs)
        self._preIntensitySpikesTimes.setIndPointsLocs(locs=locs)

    def setKernels(self, kernels):
        self._preIntensityQuadTimes.setKernels(kernels=kernels)
        self._preIntensitySpikesTimes.setKernels(kernels=kernels)

    def setInitialParams(self, initial_params):
        self._preIntensityQuadTimes.setInitialParams(initial_params=initial_params)
        self._preIntensitySpikesTimes.setInitialParams(initial_params=initial_params)

    def setELLCalculationParams(self, eLLCalculationParams):
        # self._preIntensityQuadTimes.setTimes(times=eLLCalculationParams["leg_quad_points"])
        self._legQuadWeights = eLLCalculationParams["leg_quad_weights"]

    @abstractmethod
    def _getELinkValues(self, eMean, eVar):
        pass

    @abstractmethod
    def _getELogLinkValues(self, eMean, eVar):
        pass

class PointProcessELLExpLink(PointProcessELL):
    def __init__(self, preIntensityQuadTimes, preIntensitySpikesTimes):
        super().__init__(preIntensityQuadTimes=preIntensityQuadTimes,
                         preIntensitySpikesTimes=preIntensitySpikesTimes,
                         linkFunction=jnp.exp)

    def _getELinkValues(self, eMean, eVar):
        # eLinkValues[r] \in nQuadLeg x nNeurons
        n_trials = len(eMean)
        eLinkValues = [self._linkFunction(eMean[r]+0.5*eVar[r])
                       for r in range(n_trials)]
        return eLinkValues

    def _getELogLinkValues(self, eMean, eVar):
        # eLogLink = jnp.cat([jnp.squeeze(input=eMean[trial]) for trial in range(len(eMean))])
        # eLogLink = jnp.cat([eMean[r] for r in range(len(eMean))])
        eLogLink = jnp.concatenate([eMean[r] for r in range(len(eMean))])
        return eLogLink

class PointProcessELLQuad(PointProcessELL):

    def __init__(self, preIntensityQuadTimes, preIntensitySpikesTimes, linkFunction):
        super().__init__(preIntensityQuadTimes=preIntensityQuadTimes,
                         preIntensitySpikesTimes=preIntensitySpikesTimes,
                        linkFunction=linkFunction)

    def setELLCalculationParams(self, eLLCalculationParams):
        super().setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
        self._hermQuadWeights = eLLCalculationParams["hermQuadWeights"]
        self._hermQuadPoints = eLLCalculationParams["hermQuadPoints"]

    def _getELinkValues(self, eMean, eVar):
        n_trials = len(eMean)
        # aux2[r] \in  nQuadLeg x nNeurons
        aux2 = [jnp.sqrt(2*eVar[r]) for r in range(n_trials)]
        # aux3 \in nTrials x nQuadLeg x nNeurons x nQuadLeg
        aux3 = jnp.einsum('ijk,l->ijkl', aux2, jnp.squeeze(self._hermQuadPoints))
        # aux4 \in nQuad x nQuadLeg x nTrials x nQuadLeg
        aux4 = jnp.add(input=aux3, other=eMean.unsqueeze(dim=3))
        # aux5 \in nQuad x nQuadLeg x nTrials x nQuadLeg
        aux5 = self._linkFunction(aux4)
        # intval \in  nTrials x nQuadHerm x nNeurons
        eLinkValues = jnp.einsum('ijkl,l->ijk', aux5, self._hermQuadWeights.squeeze())
        return eLinkValues

    def _getELogLinkValues(self, eMean, eVar):
        # log_link = cellvec(cellfun(@(x,y) log(m.link(x + sqrt(2*y).* m.xxHerm'))*m.wwHerm,mu_h_Spikes,var_h_Spikes,'uni',0));
        # aux1[trial] \in nSpikes[trial]
        aux1 = [2*eVar[trial] for trial in range(len(eVar))]
        # aux2[trial] \in nSpikes[trial] x nQuadLeg
        aux2 = [jnp.einsum('i,j->ij', aux1[trial].squeeze(), self._hermQuadPoints.squeeze()) for trial in range(len(aux1))]
        # aux3[trial] \in nSpikes[trial] x nQuadLeg
        aux3 = [jnp.add(input=aux2[trial],
                          other=jnp.unsqueeze(input=eMean[trial], dim=1))
                for trial in range(len(aux2))]
        # aux4[trial] \in nSpikes[trial] x nQuadLeg
        aux4 = [jnp.log(input=self._linkFunction(aux3[trial])) for trial in range(len(aux3))]
        # aux5[trial] \in nSpikes[trial] x 1
        aux5 = [jnp.tensordot(a=aux4[trial], b=self._hermQuadWeights, dims=([1], [0])) for trial in range(len(aux4))]
        # eLogLinkValues = jnp.cat(tensors=aux5)
        eLogLinkValues = jnp.concatenate(tensors=aux5)
        return eLogLinkValues


class PoissonELL(ExpectedLogLikelihood):

    def setELLCalculationParams(self, eLLCalculationParams):
        times = eLLCalculationParams["binTimes"]
        self._binWidth = times[0,1,0]-times[0,0,0]
        self._preIntensityQuadTimes.setTimes(times=times)

    def computePosteriorOnLatentsStats(self):
        answer = self._preIntensityQuadTimes.computePosteriorOnLatentsStats()
        return answer

    def evalSumAcrossTrialsAndNeurons(self, posteriorOnLatentsStats=None):
        eMean, eVar= self._preIntensityQuadTimes.\
            computeMeansAndVars(posteriorOnLatentsStats=posteriorOnLatentsStats)
        eLinkValues, eLogLinkValues = \
                self._getELinkAndELogLinkValues(eMean=eMean, eVar=eVar)
        sELLTerm1 = self._binWidth*eLinkValues.sum()
        # sELLTerm2 = (self._measurements*eLogLinkValues.permute(0, 2, 1)).sum()
        sELLTerm2 = (self._measurements*eLogLinkValues).sum()
        answer = -sELLTerm1+sELLTerm2
        return answer

    def buildKernelsMatrices(self):
        self._preIntensityQuadTimes.buildKernelsMatrices()

    def computeSVPostOnLatentsStats(self):
        answer = self._preIntensityQuadTimes.computeSVPostOnLatentsStats()
        return answer

    def setMeasurements(self, measurements):
        # measurements \in nTrials x nNeurons x maxNBins
        self._measurements = measurements

    def setIndPointsLocs(self, locs):
        self._preIntensityQuadTimes.setIndPointsLocs(locs=locs)
        self._preIntensitySpikesTimes.setIndPointsLocs(locs=locs)

    def setKernels(self, kernels):
        self._preIntensityQuadTimes.\
            setKernels(kernels=kernels)

    def setInitialParams(self, initial_params):
        self._preIntensityQuadTimes.\
            setInitialParams(initial_params=initial_params)

    @abstractmethod
    def _getELinkAndELogLinkValues(self, eMean, eVar):
        pass

class PoissonELLExpLink(PoissonELL):

    def __init__(self, preIntensityQuadTimes):
        super().__init__(preIntensityQuadTimes=preIntensityQuadTimes,
                         linkFunction=jnp.exp)

    def _getELinkAndELogLinkValues(self, eMean, eVar):
        # intval \in nTrials x nBins x nNeurons
        eLinkValues = self._linkFunction(eMean+0.5*eVar)
        eLogLinkValues = eMean
        return eLinkValues, eLogLinkValues

class PoissonELLQuad(PoissonELL):

    def __init__(self, preIntensityQuadTimes, linkFunction):
        super().__init__(preIntensityQuadTimes=preIntensityQuadTimes,
                         linkFunction=linkFunction)

    def _getELinkAndELogLinkValues(self, eMean, eVar):
        # intval = permute(mtimesx(m.wwHerm',permute(m.link(qHmeanAtQuad + sqrt(2*qHMVarAtQuad).*permute(m.xxHerm,[2 3 4 1])),[4 1 2 3])),[2 3 4 1]);

        # aux2 \in  nTrials x maxNBins x nNeurons
        aux2 = jnp.sqrt(2*eVar)
        # aux3 \in nTrials x maxNBins x nNeurons x nQuadLeg
        aux3 = jnp.einsum('ijk,l->ijkl', aux2, jnp.squeeze(self._hermQuadPoints))
        # aux4 \in maxNTrials x maxNBins x nNeurons x nQuadLeg
        aux4 = jnp.add(input=aux3, other=eMean.unsqueeze(dim=3))
        # aux5a \in maxNTrials x maxNBins x nNeurons x nQuadLeg
        aux5a = self._linkFunction(aux4)
        aux5b = aux5a.log()
        # intval \in  maxNTrials x maxNBins x nQuadLeg
        eLinkValues = jnp.einsum('ijkl,l->ijk', aux5b, self._hermQuadWeights.squeeze())
        eLoglinkValues = jnp.einsum('ijkl,l->ijk', aux5a, self._hermQuadWeights.squeeze())
        return eLinkValues, eLogLinkValues

