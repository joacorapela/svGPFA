
import warnings
import math
import time
import jax
import jaxopt

from ..utils.miscUtils import buildCovsFromCholVecs
from . import kernelsMatricesStore, expectedLogLikelihood, svLowerBound


class EM_JAXopt:

    def init(spikesTimesArray, validSpikesTimesMask, kernels, legQuadPoints,
             legQuadWeights, reg_param):
        EM_JAXopt.indPointsLocsKMS = kernelsMatricesStore.IndPointsLocsKMS_Chol
        EM_JAXopt.indPointsLocsKMS.init(kernels=kernels)
        EM_JAXopt.quadTimesKMS = \
            kernelsMatricesStore.IndPointsLocsAndQuadTimesKMS
        EM_JAXopt.quadTimesKMS.init(kernels=kernels, t=legQuadPoints)
        EM_JAXopt.spikesTimesKMS = \
            kernelsMatricesStore.IndPointsLocsAndSpikesTimesKMS
        EM_JAXopt.spikesTimesKMS.init(kernels=kernels, t=spikesTimesArray)
        EM_JAXopt.reg_param = reg_param
        EM_JAXopt.eLL = expectedLogLikelihood.PointProcessELLExpLink
        EM_JAXopt.eLL.init(legQuadWeights=legQuadWeights,
                           validSpikesTimesMask=validSpikesTimesMask)

    def maximizeECM(params0, optim_params):
        LB0 = -EM_JAXopt._eval_func_params_as_list(params0)
        print(f"Initial LB: {LB0}")
        # breakpoint()
        params = params0

        def variationalParamsOptimFunc(variational_params, additional_params):
            value = EM_JAXopt._eval_func(
                vMean=variational_params["variational_mean"],
                vChol=variational_params["variational_chol_vecs"],
                C=additional_params["C"], d=additional_params["d"],
                kernels_params=additional_params["kernels_params"],
                ind_points_locs=additional_params["ind_points_locs"],
            )
            return value

        def preIntensityParamsOptimFunc(preIntensity_params,
                                        additional_params):
            value = EM_JAXopt._eval_func(
                vMean=additional_params["variational_mean"],
                vChol=additional_params["variational_chol_vecs"],
                C=preIntensity_params["C"], d=preIntensity_params["d"],
                kernels_params=additional_params["kernels_params"],
                ind_points_locs=additional_params["ind_points_locs"],
            )
            return value

        def kernelsParamsOptimFunc(kernels_params, additional_params):
            value = EM_JAXopt._eval_func(
                vMean=additional_params["variational_mean"],
                vChol=additional_params["variational_chol_vecs"],
                C=additional_params["C"], d=additional_params["d"],
                kernels_params=kernels_params["kernels_params"],
                ind_points_locs=additional_params["ind_points_locs"],
            )
            return value

        def indPointsLocsOptimFunc(indPoints_params, additional_params):
            value = EM_JAXopt._eval_func(
                vMean=additional_params["variational_mean"],
                vChol=additional_params["variational_chol_vecs"],
                C=additional_params["C"], d=additional_params["d"],
                kernels_params=additional_params["kernels_params"],
                ind_points_locs=indPoints_params["ind_points_locs"],
            )
            return value

        variational_solver = jaxopt.LBFGS(fun=variationalParamsOptimFunc,
                                          **optim_params["variational_params"])
        # variational_solver = jaxopt.ScipyMinimize(
        #     fun=variationalParamsOptimFunc, method="L-BFGS-B",
        #     **optim_params["variational_params"])
        variational_params = {k: params[k]
                              for k in ('variational_mean',
                                        'variational_chol_vecs')}

        preIntensity_solver = jaxopt.LBFGS(
            fun=preIntensityParamsOptimFunc,
            **optim_params["preIntensity_params"])
        # preIntensity_solver = jaxopt.ScipyMinimize(
        #     fun=preIntensityParamsOptimFunc, method="L-BFGS-B",
        #     **optim_params["preIntensity_params"])
        preIntensity_params = {k: params[k] for k in ('C', 'd')}

        kernels_solver = jaxopt.LBFGS(fun=kernelsParamsOptimFunc,
                                      **optim_params["kernels_params"])
        # kernels_solver = jaxopt.ScipyMinimize(
        #     fun=kernelsParamsOptimFunc, method="L-BFGS-B",
        #     **optim_params["kernels_params"])
        kernels_params = {"kernels_params": params["kernels_params"]}

        indPointsLocs_solver = jaxopt.LBFGS(
            fun=indPointsLocsOptimFunc, **optim_params["indpointslocs_params"])
        # indPointsLocs_solver = jaxopt.ScipyMinimize(
        #     fun=indPointsLocsOptimFunc, method="L-BFGS-B",
        #     **optim_params["indpointslocs_params"])
        indPointsLocs_params = {"ind_points_locs": params["ind_points_locs"]}

        prev_lower_bound = -math.inf
        best_lower_bound = LB0
        lower_bound_hist = [LB0]
        elapsed_time_hist = [0.0]
        start_time = time.time()
        i = 0
        while (i < optim_params["n_em_iterations"] and
               (best_lower_bound - prev_lower_bound) > optim_params["tol"]):
            prev_lower_bound = best_lower_bound
            if optim_params["variational_estimate"]:
                res = variational_solver.run(variational_params,
                                             additional_params=params)
                cur_lower_bound = -res.state.value
                # cur_lower_bound = -res.state.fun_val
                if cur_lower_bound > best_lower_bound:
                    best_lower_bound = cur_lower_bound
                    params["variational_mean"] = res.params["variational_mean"]
                    params["variational_chol_vecs"] = \
                        res.params["variational_chol_vecs"]
                else:
                    warnings.warn(
                        f"Iteration {i}, cur_lower_bound={cur_lower_bound} is "
                        f"lower than best_lower_bound={best_lower_bound}. "
                        f"Not updating parameters"
                    )
                print(f"Iteration {i}, variational step, "
                      f"LB={best_lower_bound}")
                # breakpoint()

            if i > 0 and optim_params["preIntensity_estimate"]:
                res = preIntensity_solver.run(preIntensity_params,
                                              additional_params=params)
                cur_lower_bound = -res.state.value
                # cur_lower_bound = -res.state.fun_val
                if cur_lower_bound > best_lower_bound:
                    best_lower_bound = cur_lower_bound
                    params["C"] = res.params["C"]
                    params["d"] = res.params["d"]
                else:
                    warnings.warn(
                        f"Iteration {i}, cur_lower_bound={cur_lower_bound} is "
                        f"lower than best_lower_bound={best_lower_bound}. "
                        f"Not updating parameters"
                    )
                print(f"Iteration {i}, preIntensity step, "
                      f"LB={best_lower_bound}")
                # breakpoint()

            if optim_params["kernels_estimate"]:
                res = kernels_solver.run(kernels_params,
                                         additional_params=params)
                cur_lower_bound = -res.state.value
                # cur_lower_bound = -res.state.fun_val
                if cur_lower_bound > best_lower_bound:
                    best_lower_bound = cur_lower_bound
                    params["kernels_params"] = res.params["kernels_params"]
                else:
                    warnings.warn(
                        f"Iteration {i}, cur_lower_bound={cur_lower_bound} is "
                        f"lower than best_lower_bound={best_lower_bound}. "
                        f"Not updating parameters"
                    )
                print(f"Iteration {i}, kernels step, LB={best_lower_bound}")
                # breakpoint()

            if optim_params["indpointslocs_estimate"]:
                res = indPointsLocs_solver.run(indPointsLocs_params,
                                               additional_params=params)
                cur_lower_bound = -res.state.value
                # cur_lower_bound = -res.state.fun_val
                if cur_lower_bound > best_lower_bound:
                    best_lower_bound = cur_lower_bound
                    params["ind_points_locs"] = res.params["ind_points_locs"]
                else:
                    warnings.warn(
                        f"Iteration {i}, cur_lower_bound={cur_lower_bound} is "
                        f"lower than best_lower_bound={best_lower_bound}. "
                        f"Not updating parameters"
                    )
                print(f"Iteration {i}, indPointsLocs step, "
                      f"LB={best_lower_bound}")
                # breakpoint()
            lower_bound_hist.append(best_lower_bound)
            elapsed_time_hist.append(time.time()-start_time)
            i += 1
        answer = dict(params=params, lower_bound_hist=lower_bound_hist,
                      elapsed_time_hist=elapsed_time_hist)
        return answer

    def maximize(params0, optim_params):
        print(f"Initial LB: {-EM_JAXopt._eval_func_params_as_list(params0)}")
        solver = jaxopt.LBFGS(fun=EM_JAXopt._eval_func_params_as_list,
                              **optim_params)
        res = solver.run(params0)
        return res

    def maximizeInSteps(params0, optim_params):
        solver = jaxopt.LBFGS(fun=EM_JAXopt._eval_func_params_as_list,
                              **optim_params)
        print("About to call solver.init_state(params)")
        state = solver.init_state(params0)
        print("Called solver.init_state(params) done")

        params = params0
        for step in range(optim_params["maxiter"]):
            params, state = solver.update(params=params, state=state)
            lower_bound = -state.value
            print(f"Iteration {step}: {lower_bound}")
        return params, state

    def maximize_jaxopt_scipyMinimize(params0, optim_params):
        print(f"Initial LB: {EM_JAXopt._eval_func_params_as_list(params0)}")

        def mycallback(params):
            lb = -EM_JAXopt._eval_func_params_as_list(params)
            print(f"lower bound: {lb}")

        solver = jaxopt.ScipyMinimize(fun=EM_JAXopt._eval_func_params_as_list,
                                      method="L-BFGS-B",
                                      callback=mycallback,
                                      **optim_params)
        res = solver.run(params0)
        return res

    def _eval_func_params_as_list(params):
        vMean = params["variational_mean"]
        vChol = params["variational_chol_vecs"]
        C = params["C"]
        d = params["d"]
        kernels_params = params["kernels_params"]
        ind_points_locs = params["ind_points_locs"]
        answer = EM_JAXopt._eval_func(vMean=vMean, vChol=vChol, C=C, d=d,
                                      kernels_params=kernels_params,
                                      ind_points_locs=ind_points_locs)
        return answer

    @jax.jit
    def _eval_func(vMean, vChol, C, d, kernels_params, ind_points_locs):

        Kzz, Kzz_cho = EM_JAXopt.indPointsLocsKMS.buildKernelsMatrices(
            kernels_params=kernels_params, ind_points_locs=ind_points_locs,
            reg_param=EM_JAXopt.reg_param)
        Ktz_quad = EM_JAXopt.quadTimesKMS.buildKernelsMatrices(
            kernels_params=kernels_params, ind_points_locs=ind_points_locs)
        Ktz_spikes = EM_JAXopt.spikesTimesKMS.buildKernelsMatrices(
            kernels_params=kernels_params, ind_points_locs=ind_points_locs)
        vCov = buildCovsFromCholVecs(vChol)
        svlb = svLowerBound.SVLowerBound
        answer = -svlb.eval(vMean=vMean, vCov=vCov, C=C, d=d, Kzz=Kzz,
                            Kzz_cho=Kzz_cho, KtzQuad=Ktz_quad,
                            KtzSpikes=Ktz_spikes, KttDiag=1.0)
        print(f"LB: {-answer}")
        return answer


class TerminationInfo:
    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message


class ErrorTerminationInfo(TerminationInfo):
    def __init__(self, message, error, stack_trace):
        super().__init__(message=message)
        self._error = error
        self._stack_trace = stack_trace

    @property
    def error(self):
        return self._error

    @property
    def stack_trace(self):
        return self._stack_trace
