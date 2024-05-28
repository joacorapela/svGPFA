
import jax
import jax.numpy as jnp

from . import preIntensity


class PointProcessELLExpLink:

    def init(legQuadWeights, validSpikesTimesMask):
        PointProcessELLExpLink.legQuadWeights = legQuadWeights
        PointProcessELLExpLink.validSpikesTimesMask = validSpikesTimesMask

    def evalSumAcrossTrialsAndNeurons(vMean, vCov, C, d, Kzz, Kzz_inv,
                                      KtzQuad, KtzSpikes, KttDiag):
        # vMean \in n_latents x n_trials x n_ind_points
        # vCov \in
        #  n_latents x n_trials x n_ind_points x n_ind_points
        # C \in n_neurons x n_trials
        # d \in n_neurons x 1
        # Kzz \in n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz_inv \in n_latents x n_trials x n_ind_points x n_ind_points
        # KtzQuad \in n_latents x n_trials x n_quad x n_ind_points
        # KtzSpikes \in n_latents x n_trials x n_spikes x n_ind_points
        # KttDiag \in Real
        # return Real

        # qHMuQuad \in n_trials x n_neurons x n_quad
        # qHVarQuad \in n_trials x n_neurons x n_quad
        qHMuQuad = preIntensity.LinearPreIntensityQuad.computeMeans(
            vMean=vMean, C=C, d=d, Kzz_inv=Kzz_inv, Ktz=KtzQuad)
        qHVarQuad = preIntensity.LinearPreIntensityQuad.computeVars(
            vCov=vCov, C=C, Kzz=Kzz, Kzz_inv=Kzz_inv, Ktz=KtzQuad,
            KttDiag=KttDiag)
        # qHMuSpikes \in n_trials x n_neurons x n_spikes
        qHMuSpikes = preIntensity.LinearPreIntensityQuad.computeMeans(
            vMean=vMean, C=C, d=d, Kzz_inv=Kzz_inv, Ktz=KtzSpikes)
        # eLinkValues \in n_trials x n_quad_leg x n_neurons
        eLinkValues = PointProcessELLExpLink._getELinkValues(
            qHMu=qHMuQuad, qHVar=qHVarQuad)
        eLogLinkValues = jnp.where(PointProcessELLExpLink.validSpikesTimesMask, qHMuSpikes, 0.0)
        # legQuadWeights \in nTrials x nQuad x 1
        # eLinkValues \in  nTrials x nQuad x nNeurons
        # aux1 \in  nTrials x nNeurons x 1

        def computeAux1(legQuadWeights, eLinkValues):
            # legQuadWeights \in nQuad x 1
            # eLinkValues \in nNeurons x nQuad
            # answer \in nNeurons x 1
            answer = jnp.matmul(eLinkValues, legQuadWeights)
            return answer

        computeAux1_trials = jax.vmap(computeAux1)
        # aux1_trials \in n_trials x n_neurons x 1
        aux1_trials = computeAux1_trials(PointProcessELLExpLink.legQuadWeights, eLinkValues)
        # aux1 = [jnp.matmul(legQuadWeights[r].T, eLinkValues[r]) for r in range(n_trials)]
        # sELLTerm1 = jnp.sum(jnp.cat([aux1[r] for r in range(nTrials)]))
        # sELLTerm1 = jnp.sum(jnp.concatenate( [aux1[r] for r in range(n_trials)]))
        sELLTerm1 = jnp.sum(aux1_trials)
        sELLTerm2 = jnp.sum(eLogLinkValues)
        answer = -sELLTerm1+sELLTerm2
        return answer

    def _getELinkValues(qHMu, qHVar, linkFunction=jnp.exp):
        # qHMu \in n_trials x n_neurons x n_quad
        # qHVar \in n_trials x n_neurons x n_quad
        # eLinkValues \in n_trials x n_quad_leg x n_neurons
        eLinkValues = linkFunction(qHMu + 0.5 * qHVar)
        return eLinkValues
