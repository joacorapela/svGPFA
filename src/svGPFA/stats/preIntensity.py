
import jax
import jax.numpy as jnp

from . import posteriorOnLatents


class LinearPreIntensityQuad:

    def computeMeansAndVars(vMean, vCov, C, d, Kzz, Kzz_inv, Ktz, KttDiag):
        qKMu, qKVar = \
            posteriorOnLatents.PosteriorOnLatentsQuad.computeMeansAndVars(
                vMean=vMean, vCov=vCov, Kzz=Kzz, Kzz_inv=Kzz_inv, Ktz=Ktz,
                KttDiag=KttDiag)
        qHMu, qHVar = LinearPreIntensityQuad.\
            _computeMeansAndVarsGivenPosteriorOnLatentsStats(
                qKMu=qKMu, qKVar=qKVar, C=C, d=d)
        return qHMu, qHVar

    @jax.jit
    def _computeMeansAndVarsGivenPosteriorOnLatentsStats(qKMu, qKVar, C, d):
        # qKMu \in n_latents x n_trials x n_quad
        # qKVar \in n_latents x n_trials x n_quad
        # qHMu \in n_trials x n_neurons x n_quad
        # qHVar \in n_trials x n_neurons x n_quad

        def posteriorOnMeans(qKMu, C, d):
            # qKMu \in n_latents x n_quad
            # C \in n_neurons x n_latents
            # d \in n_neurons x 1
            # qHMu \ in n_neurons x n_quad
            qHMu = C @ qKMu[:, :] + d
            return qHMu

        posteriorOnMeans_vmTrials = jax.vmap(posteriorOnMeans, (1, None, None))
        qHMu = posteriorOnMeans_vmTrials(qKMu, C, d)

        def posteriorOnVars(qKVar, C):
            # qKVar \in n_latents x n_quad
            # C \in n_neurons x n_latents
            # qHVar \ in n_neurons x n_quad
            qHVar = C**2 @ qKVar
            return qHVar

        posteriorOnVars_vmTrials = jax.vmap(posteriorOnVars, (1, None))
        qHVar = posteriorOnVars_vmTrials(qKVar, C)

        return qHMu, qHVar


class LinearPreIntensitySpikes:

    def init(neuronForSpikeIndex):
        LinearPreIntensitySpikes.neuronForSpikeIndex = neuronForSpikeIndex

    def computeMeansAndVars(vMean, vCov, C, d, Kzz, Kzz_inv, Ktz, KttDiag):
        qKMu, qKVar = \
            posteriorOnLatents.PosteriorOnLatentsSpikes.computeMeansAndVars(
                vMean=vMean, vCov=vCov, Kzz=Kzz, Kzz_inv=Kzz_inv, Ktz=Ktz,
                KttDiag=KttDiag)
        qHMu, qHVar = LinearPreIntensitySpikes.\
            _computeMeansAndVarsGivenPosteriorOnLatentsStats(
                qKMu=qKMu, qKVar=qKVar, C=C, d=d)
        return qHMu, qHVar

    def _computeMeansAndVarsGivenPosteriorOnLatentsStats(qKMu, qKVar, C, d):
        # qKMu[r], qKVar[r] \in n_spikes_all_neurons[r] x n_latents
        # qHMu[r], qHVar[r] \in nSpikesFromAllNeuronsInTrial[r]
        n_trials = len(LinearPreIntensitySpikes.neuronForSpikeIndex)
        qHMu = [[None] for tr in range(n_trials)]
        qHVar = [[None] for tr in range(n_trials)]
        for r in range(n_trials):
            qHMu[r] = (jnp.sum(qKMu[r] *
                               C[LinearPreIntensitySpikes.neuronForSpikeIndex[r], :],
                               axis=1)
                       + d[LinearPreIntensitySpikes.neuronForSpikeIndex[r]].squeeze())
            qHVar[r] = jnp.sum(qKVar[r] *
                               C[LinearPreIntensitySpikes.neuronForSpikeIndex[r], :]**2,
                               axis=1)
        return qHMu, qHVar
