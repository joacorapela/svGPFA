
import pdb
import abc
import torch

class SVEmbedding(abc.ABC):

    def __init__(self, svPosteriorOnLatents):
        self._svPosteriorOnLatents = svPosteriorOnLatents

    def computeMeansAndVars(self, svPosteriorOnLatentsStats=None):
        if svPosteriorOnLatentsStats is None:
            svPosteriorOnLatentsStats = \
                self._svPosteriorOnLatents.computeMeansAndVars()
        means, vars = self._computeMeansAndVarsGivenSVPosteriorOnLatentsStats(
            means=svPosteriorOnLatentsStats[0],
            vars=svPosteriorOnLatentsStats[1])
        return means, vars

    def computeSVPosteriorOnLatentsStats(self):
        return self._svPosteriorOnLatents.computeMeansAndVars()

    def buildKernelsMatrices(self):
        self._svPosteriorOnLatents.buildKernelsMatrices()

    @abc.abstractmethod
    def setInitialParams(self, initial_params):
        pass

    def setIndPointsLocs(self, indPointsLocs):
        self._svPosteriorOnLatents.setIndPointsLocs(indPointsLocs=indPointsLocs)

    def setKernels(self, kernels):
        self._svPosteriorOnLatents.setKernels(kernels=kernels)

    def setTimes(self, times):
        self._svPosteriorOnLatents.setTimes(times=times)

    @abc.abstractmethod
    def getParams(self):
        pass

    def getSVPosteriorOnIndPointsParams(self):
        return self._svPosteriorOnLatents.getSVPosteriorOnIndPointsParams()

    def getIndPointsLocs(self):
        return self._svPosteriorOnLatents.getIndPointsLocs()

    def getKernels(self):
        return self._svPosteriorOnLatents.getKernels()

    def getKernelsParams(self):
        return self._svPosteriorOnLatents.getKernelsParams()

    @abc.abstractmethod
    def _computeMeansAndVarsGivenSVPosteriorOnLatentsStats(self, means, vars):
        pass

class LinearSVEmbedding(SVEmbedding):

    def setInitialParams(self, initial_params):
        svEmbeddingInitialParams = initial_params["embedding"]
        self._C = svEmbeddingInitialParams["C0"]
        self._d = svEmbeddingInitialParams["d0"]
        svPosteriorOnLatentsInitialParams = initial_params["posterior_on_latents"]
        self._svPosteriorOnLatents.setInitialParams(
            initial_params=svPosteriorOnLatentsInitialParams)

    def sample(self, times, nudget=1e-3):
        latentsSamples, _, _ = self._svPosteriorOnLatents.sample(times=times,
                                                                 nudget=nudget)
        answer = [self._C.matmul(latentsSamples[r])+self._d for r in range(len(latentsSamples))]
        return answer

    def getParams(self):
        return [self._C, self._d]

class LinearSVEmbeddingAllTimes(LinearSVEmbedding):

    def _computeMeansAndVarsGivenSVPosteriorOnLatentsStats(self, means, vars):
        # emb_post_mean, emb_post_var \in nTrials x nQuad x nNeurons
        emb_post_mean = torch.matmul(means, torch.t(self._C)) + torch.reshape(input=self._d, shape=(1, 1, len(self._d))) # using broadcasting
        emb_post_var = torch.matmul(vars, (torch.t(self._C))**2)
        return emb_post_mean, emb_post_var

    def predictLatents(self, times):
        return self._svPosteriorOnLatents.predict(times=times)

    def predict(self, times):
        qKMu, qKVar = self._svPosteriorOnLatents.predict(times=times)
        answer = self._computeMeansAndVarsGivenSVPosteriorOnLatentsStats(means=qKMu, vars=qKVar)
        return answer

#     def computeMeans(self, times):
#         qKMu = self._svPosteriorOnLatents.computeMeans(times=times)
#         emb_post_mean = torch.matmul(qKMu, torch.t(self._C)) + torch.reshape(input=self._d, shape=(1, 1, len(self._d))) # using broadcasting
#         return emb_post_mean
# 
    def computeMeansAndVarsAtTimes(self, times):
        qKMu, qKVar = self._svPosteriorOnLatents.computeMeansAndVarsAtTimes(times=times)
        answer = self._computeMeansAndVarsGivenSVPosteriorOnLatentsStats(means=qKMu, vars=qKVar)
        return answer

    def setPriorCovRegParam(self, priorCovRegParam):
        self._svPosteriorOnLatents.setPriorCovRegParam(priorCovRegParam=priorCovRegParam)

class LinearSVEmbeddingAllTimesWithParamsGettersAndSetters(LinearSVEmbeddingAllTimes):
    def get_flattened_params(self):
        flattened_params = []
        flattened_params.extend(self._C.flatten().tolist())
        flattened_params.extend(self._d.flatten().tolist())
        return flattened_params

    def get_flattened_params_grad(self):
        flattened_params_grad = []
        flattened_params_grad.extend(self._C.grad.flatten().tolist())
        flattened_params_grad.extend(self._d.grad.flatten().tolist())
        return flattened_params_grad

    def set_params_from_flattened(self, flattened_params):
        flattened_param = flattened_params[:self._C.numel()]
        self._C = torch.tensor(flattened_param, dtype=torch.double).reshape(self._C.shape)
        flattened_params = flattened_params[self._C.numel():]
        flattened_param = flattened_params[:self._d.numel()]
        self._d = torch.tensor(flattened_param, dtype=torch.double).reshape(self._d.shape)
        flattened_param = flattened_params[self._d.numel():]

    def set_params_requires_grad(self, requires_grad):
        self._C.requires_grad = requires_grad
        self._d.requires_grad = requires_grad

class LinearSVEmbeddingAssocTimes(LinearSVEmbedding):

    def setNeuronForSpikeIndex(self, neuronForSpikeIndex):
        self._neuronForSpikeIndex = neuronForSpikeIndex

    def _computeMeansAndVarsGivenSVPosteriorOnLatentsStats(self, means, vars):
        nTrials = len(self._neuronForSpikeIndex)
        emb_post_mean = [[None] for tr in range(nTrials)]
        emb_post_var = [[None] for tr in range(nTrials)]
        for trialIndex in range(nTrials):
            emb_post_mean[trialIndex] = torch.sum(means[trialIndex]*self._C[(self._neuronForSpikeIndex[trialIndex]).tolist(),:], dim=1)+self._d[(self._neuronForSpikeIndex[trialIndex]).tolist()].squeeze()
            emb_post_var[trialIndex] = torch.sum(vars[trialIndex]*(self._C[(self._neuronForSpikeIndex[trialIndex]).tolist(),:])**2, dim=1)
        return emb_post_mean, emb_post_var

