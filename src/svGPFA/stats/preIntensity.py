
import jax
import jax.numpy as jnp

from . import posteriorOnLatents


class LinearPreIntensityQuad:

    def computeMeans(vMean, C, d, Kzz_inv, Ktz):
        # vMean \in n_latents x n_trials x n_ind_points
        # C \in n_neurons x n_trials
        # d \in n_neurons x 1
        # Kzz_inv \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz \in n_latents x n_trials x n_quad x n_ind_points
        # return n_trials x n_neurons x n_quad
        qKMu = posteriorOnLatents.PosteriorOnLatentsQuad.computeMeans(
            vMean=vMean, Kzz_inv=Kzz_inv, Ktz=Ktz)
        qHMu = LinearPreIntensityQuad.\
            _computeMeansGivenPosteriorOnLatentsStats(
                qKMu=qKMu, C=C, d=d)
        return qHMu

    def computeVars(vCov, C, Kzz, Kzz_inv, Ktz, KttDiag):
        # vMean \in n_latents x n_trials x n_ind_points
        # vCov \in
        #  n_latents x n_trials x n_ind_points x n_ind_points
        # C \in n_neurons x n_trials
        # d \in n_neurons x 1
        # Kzz \in n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz_inv \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz \in n_latents x n_trials x n_quad x n_ind_points
        # KttDiag \in Reals
        # return n_trials x n_neurons x n_quad
        qKVar = posteriorOnLatents.PosteriorOnLatentsQuad.computeVars(
            vCov=vCov, Kzz=Kzz, Kzz_inv=Kzz_inv, Ktz=Ktz, KttDiag=KttDiag)
        qHVar = LinearPreIntensityQuad._computeVarsGivenPosteriorOnLatentsStats(
            qKVar=qKVar, C=C)
        return qHVar

    @jax.jit
    def _computeMeansGivenPosteriorOnLatentsStats(qKMu, C, d):
        # qKMu \in n_latents x n_trials x n_quad | n_spikes_allNeurons_per_trial
        # qHMu \in n_trials x n_neurons x n_quad | n_spikes_allNeurons_per_trial
        # C \in n_neurons x n_trials
        # d \in n_neurons x 1
        # answer n_trials x n_neurons x n_quad

        def posteriorOnMeans(qKMu, C, d):
            # qKMu \in n_latents x n_quad
            # C \in n_neurons x n_latents
            # d \in n_neurons x 1
            # qHMu \ in n_neurons x n_quad
            qHMu = C @ qKMu[:, :] + d
            return qHMu

        posteriorOnMeans_vmTrials = jax.vmap(posteriorOnMeans, (1, None, None))
        qHMu = posteriorOnMeans_vmTrials(qKMu, C, d)
        return qHMu

    @jax.jit
    def _computeVarsGivenPosteriorOnLatentsStats(qKVar, C):
        # qKVar \in n_latents x n_trials x n_quad | n_spikes_per_trial
        # qHVar \in n_trials x n_neurons x n_quad | n_spikes_per_trial
        # C \in n_neurons x n_latents
        # answer n_trials x n_neurons x n_quad

        def posteriorOnVars(qKVar, C):
            # qKVar \in n_latents x n_quad
            # C \in n_neurons x n_latents
            # qHVar \ in n_neurons x n_quad
            qHVar = C**2 @ qKVar
            return qHVar

        posteriorOnVars_vmTrials = jax.vmap(posteriorOnVars, (1, None))
        qHVar = posteriorOnVars_vmTrials(qKVar, C)

        return qHVar


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
