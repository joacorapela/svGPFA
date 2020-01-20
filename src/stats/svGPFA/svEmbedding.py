
import pdb
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .kernelMatricesStore import KernelMatricesStore

class SVEmbedding(ABC, nn.Module):

    def __init__(self, svPosteriorOnLatents):
        super(SVEmbedding, self).__init__()
        self._svPosteriorOnLatents = svPosteriorOnLatents

    def computeMeansAndVars(self, svPosteriorOnLatentsStats=None):
        if svPosteriorOnLatentsStats is None:
            svPosteriorOnLatentsStats = \
                self._svPosteriorOnLatents.computeMeansAndVars()
        means, vars = self._getMeansAndVarsGivenSVPosteriorOnLatentsStats(
            means=svPosteriorOnLatentsStats[0],
            vars=svPosteriorOnLatentsStats[1])
        return means, vars

    def computeSVPosteriorOnLatentsStats(self):
        return self._svPosteriorOnLatents.computeMeansAndVars()

    def buildKernelsMatrices(self):
        self._svPosteriorOnLatents.buildKernelsMatrices()

    @abstractmethod
    def setInitialParams(self, initialParams):
        pass

    def setIndPointsLocs(self, indPointsLocs):
        self._svPosteriorOnLatents.setIndPointsLocs(indPointsLocs=indPointsLocs)

    def setKernels(self, kernels):
        self._svPosteriorOnLatents.setKernels(kernels=kernels)

    def setTimes(self, times):
        self._svPosteriorOnLatents.setTimes(times=times)

    @abstractmethod
    def getParams(self):
        pass

    def getSVPosteriorOnIndPointsParams(self):
        return self._svPosteriorOnLatents.getSVPosteriorOnIndPointsParams()

    def getIndPointsLocs(self):
        return self._svPosteriorOnLatents.getIndPointsLocs()

    def getKernelsParams(self):
        return self._svPosteriorOnLatents.getKernelsParams()

    @abstractmethod
    def _getMeansAndVarsGivenSVPosteriorOnLatentsStats(self, means,
                                                        vars):
        pass

class LinearSVEmbedding(SVEmbedding):

    def setInitialParams(self, initialParams):
        svEmbeddingInitialParams = initialParams["svEmbedding"]
        self._C = svEmbeddingInitialParams["C0"]
        self._d = svEmbeddingInitialParams["d0"]
        self._svPosteriorOnLatents.setInitialParams(initialParams=initialParams)

    def getParams(self):
        return [self._C, self._d]

class LinearSVEmbeddingAllTimes(LinearSVEmbedding):

    def _getMeansAndVarsGivenSVPosteriorOnLatentsStats(self, means,
                                                        vars):
        qHMu = torch.matmul(means, torch.t(self._C)) + torch.reshape(input=self._d, shape=(1, 1, len(self._d))) # using broadcasting
        qHVar = torch.matmul(vars, (torch.t(self._C))**2)
        return qHMu, qHVar

    def predictLatents(self, newTimes):
        return self._svPosteriorOnLatents.predict(newTimes=newTimes)

class LinearSVEmbeddingAssocTimes(LinearSVEmbedding):

    def setNeuronForSpikeIndex(self, neuronForSpikeIndex):
        self._neuronForSpikeIndex = neuronForSpikeIndex

    def _getMeansAndVarsGivenSVPosteriorOnLatentsStats(self, means, vars):
        nTrials = len(self._neuronForSpikeIndex)
        qHMu = [[None] for tr in range(nTrials)]
        qHVar = [[None] for tr in range(nTrials)]
        for trialIndex in range(nTrials):
            qHMu[trialIndex] = torch.sum(means[trialIndex]*self._C[(self._neuronForSpikeIndex[trialIndex]).tolist(),:], dim=1)+self._d[(self._neuronForSpikeIndex[trialIndex]).tolist()].squeeze()
            qHVar[trialIndex] = torch.sum(vars[trialIndex]*(self._C[(self._neuronForSpikeIndex[trialIndex]).tolist(),:])**2, dim=1)
        return qHMu, qHVar

