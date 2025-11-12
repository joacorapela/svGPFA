
import jax
import jax.numpy as jnp

from . import posteriorOnLatents


class LinearPreIntensity:

    def computeMeans(vMean: jax.Array, C: jax.Array, d: jax.Array,
                     Kzz_cho: jax.Array, Ktz: jax.Array) -> jax.Array:
        # vMean \in n_latents x n_trials x n_ind_points
        # C \in n_neurons x n_trials
        # d \in n_neurons x 1
        # Kzz_cho \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz \in n_latents x n_trials x n_quad x n_ind_points
        # return n_trials x n_neurons x n_quad
        qKMu = posteriorOnLatents.PosteriorOnLatents.computeMeans(
            vMean=vMean, Kzz_cho=Kzz_cho, Ktz=Ktz)
        qHMu = LinearPreIntensity._computeMeansGivenPosteriorOnLatentsStats(
            qKMu=qKMu, C=C, d=d)
        return qHMu

    def computeVars(vCov: jax.Array, C: jax.Array, Kzz: jax.Array,
                    Kzz_cho: jax.Array, Ktz: jax.Array,
                    KttDiag: float = 1.0) -> jax.Array:
        # vMean \in n_latents x n_trials x n_ind_points
        # vCov \in
        #  n_latents x n_trials x n_ind_points x n_ind_points
        # C \in n_neurons x n_trials
        # d \in n_neurons x 1
        # Kzz \in n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz_cho \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz \in n_latents x n_trials x n_quad x n_ind_points
        # KttDiag \in Reals
        # return n_trials x n_neurons x n_quad
        qKVar = posteriorOnLatents.PosteriorOnLatents.computeVars(
            vCov=vCov, Kzz=Kzz, Kzz_cho=Kzz_cho, Ktz=Ktz, KttDiag=KttDiag)
        qHVar = LinearPreIntensity._computeVarsGivenPosteriorOnLatentsStats(
            qKVar=qKVar, C=C)
        return qHVar

    def _computeMeansGivenPosteriorOnLatentsStats(qKMu: jax.Array,
                                                  C: jax.Array,
                                                  d: jax.Array) -> jax.Array:
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
            # qHMu = C @ qKMu[:, :] + d
            qHMu = C @ qKMu + d
            return qHMu

        posteriorOnMeans_vmTrials = jax.vmap(posteriorOnMeans, (1, None, None))
        qHMu = posteriorOnMeans_vmTrials(qKMu, C, d)
        return qHMu

    def _computeVarsGivenPosteriorOnLatentsStats(qKVar: jax.Array, C: jax.Array):
        # qKVar \in n_latents x n_trials x n_quad | n_spikes_per_trial
        # qHVar \in n_trials x n_neurons x n_quad | n_spikes_per_trial
        # C \in n_neurons x n_latents
        # answer n_trials x n_neurons x n_quad

        def posteriorOnVars(qKVar: jax.Array, C: jax.Array):
            # qKVar \in n_latents x n_quad
            # C \in n_neurons x n_latents
            # qHVar \ in n_neurons x n_quad
            qHVar = C**2 @ qKVar
            return qHVar

        posteriorOnVars_vmTrials = jax.vmap(posteriorOnVars, (1, None))
        qHVar = posteriorOnVars_vmTrials(qKVar, C)

        return qHVar
