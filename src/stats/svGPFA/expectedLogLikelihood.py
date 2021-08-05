
import pdb
from abc import ABC, abstractmethod
import torch
# import warnings

class ExpectedLogLikelihood(ABC):

    def __init__(self, svEmbeddingAllTimes, linkFunction):
        self._svEmbeddingAllTimes = svEmbeddingAllTimes
        self._linkFunction = linkFunction

    @abstractmethod
    def evalSumAcrossTrialsAndNeurons(self, svPosteriorOnLatentsStats=None):
        pass

    @abstractmethod
    def sampleCIFs(self):
        pass

    @abstractmethod
    def buildKernelsMatrices(self):
        pass

    @abstractmethod
    def computeSVPosteriorOnLatentsStats(self):
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
    def setInitialParams(self, initialParams):
        pass

    @abstractmethod
    def setQuadParams(self, quadParams):
        pass

    def getSVPosteriorOnIndPointsParams(self):
        return self._svEmbeddingAllTimes.getSVPosteriorOnIndPointsParams()

    def getSVEmbeddingParams(self):
        return self._svEmbeddingAllTimes.getParams()

    def computeEmbeddingsMeansAndVarsAtTimes(self, times):
        return self._svEmbeddingAllTimes.computeMeansAndVarsAtTimes(times)

    def getIndPointsLocs(self):
        return self._svEmbeddingAllTimes.getIndPointsLocs()

    def getKernelsParams(self):
        return self._svEmbeddingAllTimes.getKernelsParams()

    def predictLatents(self, newTimes):
        return self._svEmbeddingAllTimes.predictLatents(newTimes=newTimes)

    def setIndPointsLocsKMSRegEpsilon(self, indPointsLocsKMSRegEpsilon):
        self._svEmbeddingAllTimes.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)

class PointProcessELL(ExpectedLogLikelihood):
    def __init__(self, svEmbeddingAllTimes, svEmbeddingAssocTimes, linkFunction):
        super().__init__(svEmbeddingAllTimes=svEmbeddingAllTimes, linkFunction=linkFunction)
        self._svEmbeddingAssocTimes = svEmbeddingAssocTimes

    def evalSumAcrossTrialsAndNeurons(self, svPosteriorOnLatentsStats=None):
        if svPosteriorOnLatentsStats is not None:
            svPosteriorOnLatentsStatsAllTimes = \
             svPosteriorOnLatentsStats["allTimes"]
            svPosteriorOnLatentsStatsAssocTimes = \
             svPosteriorOnLatentsStats["assocTimes"]
        else:
            svPosteriorOnLatentsStatsAllTimes = None
            svPosteriorOnLatentsStatsAssocTimes = None
        eMeanAllTimes, eVarAllTimes = \
            self._svEmbeddingAllTimes.computeMeansAndVars(
                svPosteriorOnLatentsStats=svPosteriorOnLatentsStatsAllTimes)
        eMeanAssocTimes, eVarAssocTimes = \
            self._svEmbeddingAssocTimes.computeMeansAndVars(
                svPosteriorOnLatentsStats=svPosteriorOnLatentsStatsAssocTimes)
        eLinkValues = self._getELinkValues(eMean=eMeanAllTimes,
                                           eVar=eVarAllTimes)
        eLogLinkValues = self._getELogLinkValues(eMean=eMeanAssocTimes,
                                                 eVar=eVarAssocTimes)
        # self._legQuadWeights \in nTrials x nQuadHerm x 1
        # aux0 \in nTrials x 1 x nQuadHerm
        aux0 = torch.transpose(input=self._legQuadWeights, dim0=1, dim1=2)
        # eLinkValues \in  nTrials x nQuadHerm x nNeurons
        # aux1 \in  nTrials x 1 x nNeurons
        aux1 = torch.matmul(aux0, eLinkValues)
        sELLTerm1 = torch.sum(aux1)
        sELLTerm2 = torch.sum(eLogLinkValues)
        answer = -sELLTerm1+sELLTerm2
        return answer

    def sampleCIFs(self, times):
        h = self._svEmbeddingAllTimes.sample(times=times)
        nTrials = len(h)
        answer = [self._linkFunction(h[r]) for r in range(nTrials)]
        return answer

    def computeCIFsMeans(self, times):
        # h \in nTrials x times x nNeurons
        h = self._svEmbeddingAllTimes.computeMeans(times=times)
        nTrials = h.shape[0]
        nNeurons = h.shape[2]
        answer = [[self._linkFunction(h[r,:,n]) for n in range(nNeurons)] for r in range(nTrials)]
        return answer

    def buildKernelsMatrices(self):
        self._svEmbeddingAllTimes.buildKernelsMatrices()
        self._svEmbeddingAssocTimes.buildKernelsMatrices()

    def computeSVPosteriorOnLatentsStats(self):
        allTimesStats = self._svEmbeddingAllTimes.\
            computeSVPosteriorOnLatentsStats()
        assocTimesStats = self._svEmbeddingAssocTimes.\
            computeSVPosteriorOnLatentsStats()
        answer = {"allTimes": allTimesStats, "assocTimes": assocTimesStats}
        return answer

    def setMeasurements(self, measurements):

        stackedSpikeTimes, neuronForSpikeIndex = \
            self.__stackSpikeTimes(spikeTimes=measurements)
        self._svEmbeddingAssocTimes.setTimes(times=stackedSpikeTimes)
        self._svEmbeddingAssocTimes.setNeuronForSpikeIndex(neuronForSpikeIndex=
                                                            neuronForSpikeIndex)

    def __stackSpikeTimes(self, spikeTimes):
        # spikeTimes list[nTrials][nNeurons][nSpikes]
        nNeurons = len(spikeTimes)

        device = spikeTimes[0][0].device
        nTrials = len(spikeTimes)
        stackedSpikeTimes = [[] for i in range(nTrials)]
        neuronForSpikeIndex = [[] for i in range(nTrials)]
        for trialIndex in range(nTrials):
            aList = [spikeTime for neuronIndex in range(len(spikeTimes[trialIndex])) for spikeTime in spikeTimes[trialIndex][neuronIndex]]
            stackedSpikeTimes[trialIndex] = torch.tensor(aList, device=device)
            # stackedSpikeTimes[trialIndex] = torch.unsqueeze(
            #     stackedSpikeTimes[trialIndex], 1)
            aList = [neuronIndex for neuronIndex in range(len(spikeTimes[trialIndex])) for spikeTime in spikeTimes[trialIndex][neuronIndex]]
            neuronForSpikeIndex[trialIndex] = torch.tensor(
                 aList, device=device)
        return stackedSpikeTimes, neuronForSpikeIndex

    def setIndPointsLocs(self, locs):
        self._svEmbeddingAllTimes.setIndPointsLocs(locs=locs)
        self._svEmbeddingAssocTimes.setIndPointsLocs(locs=locs)

    def setKernels(self, kernels):
        self._svEmbeddingAllTimes.setKernels(kernels=kernels)
        self._svEmbeddingAssocTimes.setKernels(kernels=kernels)

    def setInitialParams(self, initialParams):
        self._svEmbeddingAllTimes.setInitialParams(initialParams=initialParams)
        self._svEmbeddingAssocTimes.setInitialParams(initialParams=initialParams)

    def setQuadParams(self, quadParams):
        self._svEmbeddingAllTimes.setTimes(times=quadParams["legQuadPoints"])
        self._legQuadWeights = quadParams["legQuadWeights"]

    @abstractmethod
    def _getELinkValues(self, eMean, eVar):
        pass

    @abstractmethod
    def _getELogLinkValues(self, eMean, eVar):
        pass

class PointProcessELLExpLink(PointProcessELL):
    def __init__(self, svEmbeddingAllTimes, svEmbeddingAssocTimes):
        super().__init__(svEmbeddingAllTimes=svEmbeddingAllTimes,
                         svEmbeddingAssocTimes=svEmbeddingAssocTimes,
                         linkFunction=torch.exp)

    def _getELinkValues(self, eMean, eVar):
        # eLinkValues \in nTrials x nQuadLeg x nNeurons
        eLinkValues = self._linkFunction(input=eMean+0.5*eVar)
        return eLinkValues

    def _getELogLinkValues(self, eMean, eVar):
        # eLogLink = torch.cat([torch.squeeze(input=eMean[trial]) for trial in range(len(eMean))])
        eLogLink = torch.cat([eMean[trial] for trial in range(len(eMean))])
        return eLogLink

    def computeExpectedCIFs(self, times):
        # h \in nTrials x times x nNeurons
        # eMean, eVar = self._svEmbeddingAllTimes.computeMeansAndVars(times=times)
        # answer = self._getELinkValues(eMean=eMean, eVar=eVar)
        # return answer

        eMean, eVar = self._svEmbeddingAllTimes.computeMeansAndVarsAtTimes(times=times)
        nTrials = eMean.shape[0]
        nNeurons = eMean.shape[2]
        answer = [[self._linkFunction(eMean[r,:,n]+0.5*eVar[r,:,n]) for n in range(nNeurons)] for r in range(nTrials)]
        return answer

class PointProcessELLQuad(PointProcessELL):

    def __init__(self, svEmbeddingAllTimes, svEmbeddingAssocTimes, linkFunction):
        super().__init__(svEmbeddingAllTimes=svEmbeddingAllTimes,
                         svEmbeddingAssocTimes=svEmbeddingAssocTimes,
                        linkFunction=linkFunction)

    def setQuadParams(self, quadParams):
        super().setQuadParams(quadParams=quadParams)
        self._hermQuadWeights = quadParams["hermQuadWeights"]
        self._hermQuadPoints = quadParams["hermQuadPoints"]

    def _getELinkValues(self, eMean, eVar):
        # aux2 \in  nTrials x nQuadLeg x nNeurons
        aux2 = torch.sqrt(2*eVar)
        # aux3 \in nTrials x nQuadLeg x nNeurons x nQuadLeg
        aux3 = torch.einsum('ijk,l->ijkl', aux2, torch.squeeze(self._hermQuadPoints))
        # aux4 \in nQuad x nQuadLeg x nTrials x nQuadLeg
        aux4 = torch.add(input=aux3, other=eMean.unsqueeze(dim=3))
        # aux5 \in nQuad x nQuadLeg x nTrials x nQuadLeg
        aux5 = self._linkFunction(input=aux4)
        # intval \in  nTrials x nQuadHerm x nNeurons
        eLinkValues = torch.einsum('ijkl,l->ijk', aux5, self._hermQuadWeights.squeeze())
        return eLinkValues

    def _getELogLinkValues(self, eMean, eVar):
        # log_link = cellvec(cellfun(@(x,y) log(m.link(x + sqrt(2*y).* m.xxHerm'))*m.wwHerm,mu_h_Spikes,var_h_Spikes,'uni',0));
        # aux1[trial] \in nSpikes[trial]
        aux1 = [2*eVar[trial] for trial in range(len(eVar))]
        # aux2[trial] \in nSpikes[trial] x nQuadLeg
        aux2 = [torch.einsum('i,j->ij', aux1[trial].squeeze(), self._hermQuadPoints.squeeze()) for trial in range(len(aux1))]
        # aux3[trial] \in nSpikes[trial] x nQuadLeg
        aux3 = [torch.add(input=aux2[trial],
                          other=torch.unsqueeze(input=eMean[trial], dim=1))
                for trial in range(len(aux2))]
        # aux4[trial] \in nSpikes[trial] x nQuadLeg
        aux4 = [torch.log(input=self._linkFunction(x=aux3[trial])) for trial in range(len(aux3))]
        # aux5[trial] \in nSpikes[trial] x 1
        aux5 = [torch.tensordot(a=aux4[trial], b=self._hermQuadWeights, dims=([1], [0])) for trial in range(len(aux4))]
        eLogLinkValues = torch.cat(tensors=aux5)
        return eLogLinkValues


class PoissonELL(ExpectedLogLikelihood):

    def evalSumAcrossTrialsAndNeurons(self, svPosteriorOnLatentsStats=None):
        eMean, eVar= self._svEmbeddingAllTimes.\
            computeMeansAndVars(svPosteriorOnLatentsStats=svPosteriorOnLatentsStats)
        eLink, eLogLink = self._getELinkAndELogLinkValues(eMean=eMean,
                                                           eVar=eVar)
        sELLTerm1 = self._binWidth*eLinkValues.sum()
        sELLTerm2 = (self._measurements*eLogLinkValues.permute(0, 2, 1)).sum()
        return -sELLTerm1+sELLTerm2

    def buildKernelsMatrices(self):
        self._svEmbeddingAllTimes.buildKernelsMatrices()

    def computeSVPostOnLatentsStats(self):
        answer = self._svEmbeddingAllTimes.computeSVPostOnLatentsStats()
        return answer

    def setMeasurements(self, measurements):
        # measurements \in nTrials x nNeurons x maxNBins
        self._measurements = measurements

    def setIndPointsLocs(self, locs):
        self._svEmbeddingAllTimes.setIndPointsLocs(locs=locs)
        self._svEmbeddingAssocTimes.setIndPointsLocs(locs=locs)

    def setKernels(self, kernels):
        self._svEmbeddingAllTimes.\
            setKernels(kernels=kernels)

    def setInitialParams(self, initialParams):
        self._svEmbeddingAllTimes.\
            setInitialParams(initialParams=initialParams)

    def setMiscParams(self, miscParams):
        self._binWidth = miscParams["binWidth"]

    @abstractmethod
    def _getELinkAndELogLinkValues(self, eMean, eVar):
        pass

class PoissonELLExpLink(PoissonELL):

    def __init__(self, svEmbeddingAllTime):
        super().__init__(svEmbeddingAllTime=svEmbeddingAllTime,
                         linkFunction=torch.exp)

    def _getELinkAndELogLinkValues(self, eMean, eVar):
        # intval \in nTrials x nQuadLeg x nNeurons
        eLinkValues = self._linkFunction(input=eMean+0.5*eVar)
        eLogLink = eMean
        return eLink, eLogLink

class PoissonELLQuad(PoissonELL):

    def __init__(self, svEmbeddingAllTime, linkFunction):
        super().__init__(svEmbeddingAllTime=svEmbeddingAllTime,
                         linkFunction=linkFunction)

    def _getELinkAndELogLinkValues(self, eMean, eVar):
        # intval = permute(mtimesx(m.wwHerm',permute(m.link(qHmeanAtQuad + sqrt(2*qHMVarAtQuad).*permute(m.xxHerm,[2 3 4 1])),[4 1 2 3])),[2 3 4 1]);

        # aux2 \in  nTrials x maxNBins x nNeurons
        aux2 = torch.sqrt(2*eVar)
        # aux3 \in nTrials x maxNBins x nNeurons x nQuadLeg
        aux3 = torch.einsum('ijk,l->ijkl', aux2, torch.squeeze(self._hermQuadPoints))
        # aux4 \in maxNTrials x maxNBins x nNeurons x nQuadLeg
        aux4 = torch.add(input=aux3, other=eMean.unsqueeze(dim=3))
        # aux5a \in maxNTrials x maxNBins x nNeurons x nQuadLeg
        aux5a = self._linkFunction(input=aux4)
        aux5b = aux5a.log()
        # intval \in  maxNTrials x maxNBins x nQuadLeg
        eLinkValues = torch.einsum('ijkl,l->ijk', aux5b, self._hermQuadWeights.squeeze())
        eLoglinkValues = torch.einsum('ijkl,l->ijk', aux5a, self._hermQuadWeights.squeeze())
        return eLinkValues, eLogLinkValues

