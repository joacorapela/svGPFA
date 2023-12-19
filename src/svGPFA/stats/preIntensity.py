
import pdb
import abc
import jax.numpy as jnp

class PreIntensity(abc.ABC):

    def __init__(self, posteriorOnLatents):
        self._posteriorOnLatents = posteriorOnLatents

    def computeMeansAndVars(self, variational_mean, variational_cov, C, d,
                            Kzz, Kzz_inv, Ktz, KttDiag):
        posteriorOnLatentsStats = \
            self._posteriorOnLatents.computeMeansAndVars(
                variational_mean=variational_mean,
                variational_cov=variational_cov,
                Kzz=Kzz, Kzz_inv=Kzz_inv, Ktz=Ktz, KttDiag=KttDiag)
        means, vars = self._computeMeansAndVarsGivenPosteriorOnLatentsStats(
            posterior_on_latents_means=posteriorOnLatentsStats[0],
            posterior_on_latents_vars=posteriorOnLatentsStats[1],
            C=C, d=d,
        )
        return means, vars

    def computePosteriorOnLatentsStats(self):
        return self._posteriorOnLatents.computeMeansAndVars()

    def buildKernelsMatrices(self):
        self._posteriorOnLatents.buildKernelsMatrices()

    def buildVariationalCov(self):
        self._posteriorOnLatents.buildVariationalCov()

    @abc.abstractmethod
    def setInitialParams(self, initial_params):
        pass

    def setIndPointsLocs(self, indPointsLocs):
        self._posteriorOnLatents.setIndPointsLocs(indPointsLocs=indPointsLocs)

    def setKernels(self, kernels):
        self._posteriorOnLatents.setKernels(kernels=kernels)

    def setTimes(self, times):
        self._posteriorOnLatents.setTimes(times=times)

    @abc.abstractmethod
    def getParams(self):
        pass

    def getVariationalDistParams(self):
        return self._posteriorOnLatents.getVariationalDistParams()

    def getIndPointsLocs(self):
        return self._posteriorOnLatents.getIndPointsLocs()

    def getKernels(self):
        return self._posteriorOnLatents.getKernels()

    def getKernelsParams(self):
        return self._posteriorOnLatents.getKernelsParams()

    @abc.abstractmethod
    def _computeMeansAndVarsGivenPosteriorOnLatentsStats(self, means, vars):
        pass

class LinearPreIntensity(PreIntensity):

    def setInitialParams(self, initial_params):
        svEmbeddingInitialParams = initial_params["embedding"]
        self._C = svEmbeddingInitialParams["C0"]
        self._d = svEmbeddingInitialParams["d0"]
        posteriorOnLatentsInitialParams = initial_params["posterior_on_latents"]
        self._posteriorOnLatents.setInitialParams(
            initial_params=posteriorOnLatentsInitialParams)

    def sample(self, times, nudget=1e-3):
        latentsSamples, _, _ = self._posteriorOnLatents.sample(times=times,
                                                                 nudget=nudget)
        answer = [self._C.matmul(latentsSamples[r])+self._d for r in range(len(latentsSamples))]
        return answer

    def getParams(self):
        return [self._C, self._d]

class LinearPreIntensityQuadTimes(LinearPreIntensity):

    def _computeMeansAndVarsGivenPosteriorOnLatentsStats(
        self, posterior_on_latents_means, posterior_on_latents_vars, C, d):
        # means[r], vars[r] \in nQuad[r] x nLatents 
        # emb_post_mean[r], emb_post_var[r] \in nQuad[r] x nNeurons
        n_trials = len(posterior_on_latents_means)
        n_neurons = C.shape[0]
        emb_post_mean = [[] for r in range(n_trials)]
        emb_post_var = [[] for r in range(n_trials)]
        for r in range(n_trials):
            emb_post_mean[r] = (jnp.matmul(posterior_on_latents_means[r], C.T) +
                                jnp.reshape(d, (1, n_neurons))) # using broadcasting
            emb_post_var[r] = jnp.matmul(posterior_on_latents_vars[r], C.T**2)
        return emb_post_mean, emb_post_var

    def predictLatents(self, times):
        return self._posteriorOnLatents.predict(times=times)

    def predict(self, times):
        qKMu, qKVar = self._posteriorOnLatents.predict(times=times)
        answer = self._computeMeansAndVarsGivenPosteriorOnLatentsStats(means=qKMu, vars=qKVar)
        return answer

    def computeMeansAndVarsAtTimes(self, times):
        qKMu, qKVar = self._posteriorOnLatents.computeMeansAndVarsAtTimes(times=times)
        answer = self._computeMeansAndVarsGivenPosteriorOnLatentsStats(means=qKMu, vars=qKVar)
        return answer

    def setPriorCovRegParam(self, priorCovRegParam):
        self._posteriorOnLatents.setPriorCovRegParam(priorCovRegParam=priorCovRegParam)

class LinearPreIntensitySpikesTimes(LinearPreIntensity):

    def setNeuronForSpikeIndex(self, neuronForSpikeIndex):
        self._neuronForSpikeIndex = neuronForSpikeIndex

    def _computeMeansAndVarsGivenPosteriorOnLatentsStats(
        self, posterior_on_latents_means, posterior_on_latents_vars, C, d):
        # means[r], vars[r] \in nQuad[r] x nLatents 
        # emb_post_mean[r], emb_post_var[r] \in nSpikesFromAllNeuronsInTrial[r]
        n_trials = len(self._neuronForSpikeIndex)
        emb_post_mean = [[None] for tr in range(n_trials)]
        emb_post_var = [[None] for tr in range(n_trials)]
        for r in range(n_trials):
#             emb_post_mean[r] = jnp.sum(posterior_on_latents_means[r]*C[(self._neuronForSpikeIndex[r]).tolist(),:], axis=1)+d[(self._neuronForSpikeIndex[r]).tolist()].squeeze()
#             emb_post_var[r] = torch.sum(posterior_on_latents_vars[r]*(C[(self._neuronForSpikeIndex[r]).tolist(),:])**2, dim=1)
            emb_post_mean[r] = jnp.sum(posterior_on_latents_means[r]*C[self._neuronForSpikeIndex[r],:], axis=1)+d[self._neuronForSpikeIndex[r]].squeeze()
            emb_post_var[r] = jnp.sum(posterior_on_latents_vars[r]*(C[self._neuronForSpikeIndex[r],:])**2, axis=1)
        return emb_post_mean, emb_post_var

