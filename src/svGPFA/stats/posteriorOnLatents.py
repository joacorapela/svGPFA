
import jax
import jax.lax
import jax.numpy as jnp


class PosteriorOnLatents:

    def _computeA(vMean: jax.Array, Kzz_cho: jax.Array):

        # ([n_ind_points, n_ind_points], [n_ind_points]) ->
        # [n_ind_points]
        def computeA(Kzz_cho, vMean):
            a = jax.scipy.linalg.cho_solve((Kzz_cho, True), vMean)
            return a

        # ([n_trials, n_ind_points, n_ind_points],
        #  [n_trials, n_ind_points]) -> [n_trials, n_ind_points]
        computeA_vmTrials = jax.vmap(computeA, in_axes=(0, 0), out_axes=0)
        # ([n_latents, n_trials, n_ind_points, n_ind_points],
        #  [n_latents, n_trials, n_ind_points]) ->
        # [n_latents, n_trials, n_ind_points]
        computeA_vmLatents = jax.vmap(computeA_vmTrials, in_axes=(0, 0),
                                      out_axes=0)

        # A \in [n_latents x n_trials x n_ind_points]
        A = computeA_vmLatents(Kzz_cho, vMean)

        return A

    def computeMeans(vMean: jax.Array, Kzz_cho: jax.Array, Ktz: jax.Array) -> jax.Array:
        # vMean \in n_latents x n_trials x n_ind_points
        # Kzz_cho \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz \in n_latents x n_trials x (n_quad | n_spikes) x n_ind_points
        # return n_latents x n_trials x (n_quad | n_spikes)

        A = PosteriorOnLatents._computeA(vMean=vMean, Kzz_cho=Kzz_cho)

        # ([n_quad, n_ind_points], [n_ind_points]) -> [n_quad]
        def computeMean(Ktz, A):
            answer = jnp.dot(Ktz, A)
            return answer

        # ([n_trials, n_quad, n_ind_points], [n_trials, n_ind_points]) ->
        # [n_trials, n_quad]
        computeMeans_vmTrials = jax.vmap(computeMean)
        # ([n_latents,n_trials,  n_quad, n_ind_points],
        #  [n_latents, n_trials, n_ind_points]) ->
        # [n_latents, n_trials, n_quad]
        computeMeans_vmLatents = jax.vmap(computeMeans_vmTrials)
        # qKMu\in [n_latents, n_trials, n_quad]
        qKMu = computeMeans_vmLatents(Ktz, A)

        return qKMu

    def computeMeansWithEquispacedTrialsTimes(vMean: jax.Array,
                                              Kzz_cho: jax.Array,
                                              Ktz: list) -> list:
        # vMean \in n_latents x n_trials x n_ind_points
        # Kzz_cho \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz[r] \in n_latents x n_times[r] x n_ind_points
        # return answer[r] \in n_latents x n_times[r]

        n_trials = len(Ktz)
        qKMu = [None] * n_trials

        # A \in n_latents x n_trials x n_ind_points
        A = PosteriorOnLatents._computeA(vMean=vMean, Kzz_cho=Kzz_cho)

        for r in range(n_trials):
            n_times_r = Ktz[r].shape[1]

            # ([n_times[r], n_ind_points], [n_ind_points]) -> [n_times[r]]
            def computeMean(Ktz, A):
                answer = jnp.dot(Ktz, A)
                return answer

            # ([n_latents, n_times[r], n_ind_points],
            #  [n_latents, n_ind_points]) ->
            # [n_latents, n_times[r]]
            computeMeans_vmLatents = jax.vmap(computeMean)

            # qKMu[r] \in [n_latents, n_times[r]]
            qKMu[r] = computeMeans_vmLatents(Ktz[r], A[:, r, :])

        return qKMu

    def computeVars(vCov: jax.Array, Kzz: jax.Array, Kzz_cho: jax.Array,
                    Ktz: jax.Array, KttDiag: float = 1.0) -> jax.Array:
        # vCov \in
        #  n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz \in n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz_cho \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz \in n_latents x n_trials x n_quad x n_ind_points
        # KttDiag \in Reals
        # return n_latents x n_trials x n_quad

        def computeVars_once(vCov: jax.Array, Kzz: jax.Array,
                             Kzz_cho: jax.Array, Ktz: jax.Array,
                             KttDiag: float = 1.0):
            # vCov \in n_ind_points x n_ind_points
            # Kzz \in n_ind_points x n_ind_points
            # Kzz_cho \in n_ind_points x n_ind_points
            # Ktz \in n_quad x n_ind_points
            # answer n_quad

            # B \in n_ind_points x n_quad
            B = jax.scipy.linalg.cho_solve((Kzz_cho, True), Ktz.T)
            # mm1f \in n_ind_points x n_quad
            mm1f = jnp.matmul(vCov-Kzz, B)
            # aux1 \in n_ind_points x n_quad
            aux1 = B * mm1f
            # aux2 \in n_quad
            aux2 = jnp.sum(aux1, axis=0)
            # aux3 \in n_quad
            answer = KttDiag + aux2
            return answer

        # ([n_trials, n_ind_points, n_ind_points],
        #  [n_trials, n_ind_points, n_ind_points],
        # [n_trials, n_ind_points, n_ind_points],
        # [n_trials, n_quad, n_ind_points], []) ->
        # [n_trials, n_quad]
        computeVars_vmTrials = jax.vmap(computeVars_once,
                                        in_axes=(0, 0, 0, 0, None))
        # ([n_latents, n_trials, n_ind_points, n_ind_points],
        # [n_latents, n_trials, n_ind_points, n_ind_points],
        # [n_latents, n_trials, n_ind_points, n_ind_points],
        # [n_latents, n_trials, n_quad, n_ind_points], []) ->
        # [n_latents, n_trials, n_quad]
        computeVars_vmLatents = jax.vmap(computeVars_vmTrials,
                                         in_axes=(0, 0, 0, 0, None))
        qKVar = computeVars_vmLatents(vCov, Kzz, Kzz_cho, Ktz, KttDiag)
        return qKVar

    def computeVarsWithEquispacedTrialsTimes(vCov: jax.Array, Kzz: jax.Array,
                                             Kzz_cho: jax.Array, Ktz: jax.Array,
                                             KttDiag: float = 1.0) -> jax.Array:
        # vCov \in n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz \in n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz_cho \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz[r] \in n_latents x n_times[r] x n_ind_points
        # KttDiag \in Reals
        # return answer[r] \in n_latents x n_times[r]

        n_trials = len(Ktz)
        qKVar = [None] * n_trials

        for r in range(n_trials):
            n_times_r = Ktz[r].shape[1]

            def computeVars_once(vCov: jax.Array, Kzz: jax.Array,
                                 Kzz_cho: jax.Array, Ktz: jax.Array,
                                 KttDiag: float = 1.0):
                # vCov \in n_ind_points x n_ind_points
                # Kzz \in n_ind_points x n_ind_points
                # Kzz_cho \in n_ind_points x n_ind_points
                # Ktz \in n_times[r] x n_ind_points
                # answer n_quad

                # B \in n_ind_points x n_quad
                B = jax.scipy.linalg.cho_solve((Kzz_cho, True), Ktz.T)
                # mm1f \in n_ind_points x n_quad
                mm1f = jnp.matmul(vCov-Kzz, B)
                # aux1 \in n_ind_points x n_quad
                aux1 = B * mm1f
                # aux2 \in n_quad
                aux2 = jnp.sum(aux1, axis=0)
                # aux3 \in n_quad
                answer = KttDiag + aux2
                return answer

            # ([n_latents, n_ind_points, n_ind_points],
            # [n_latents, n_ind_points, n_ind_points],
            # [n_latents, n_ind_points, n_ind_points],
            # [n_latents, n_times[r], n_ind_points], []) ->
            # [n_latents, n_times[r]]
            computeVars_vmLatents = jax.vmap(computeVars_once)

            # qKVar[r] \in [n_latents, n_times[r]]
            qKVar[r] = computeVars_vmLatents(vCov=vCov[:, r, :, :],
                                             Kzz=Kzz[:, r, :, :],
                                             Kzz_cho=Kzz_cho[:, r, :, :],
                                             Ktz=Ktz[r])
        return qKVar
