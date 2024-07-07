
import sys
import abc
import io
import time
import traceback
import math
import copy
import pickle
import numpy as np
import jax
import jaxopt
import scipy.optimize

from ..utils.miscUtils import buildCovsFromCholVecs
from . import kernelsMatricesStore, expectedLogLikelihood, svLowerBound

class EM_JAXopt:

    def init(spikesTimesArray, validSpikesTimesMask, kernels, legQuadPoints,
             legQuadWeights, reg_param):
        EM_JAXopt.indPointsLocsKMS = kernelsMatricesStore.IndPointsLocsKMS_Chol
        EM_JAXopt.indPointsLocsKMS.init(kernels=kernels)
        EM_JAXopt.quadTimesKMS = kernelsMatricesStore.IndPointsLocsAndQuadTimesKMS
        EM_JAXopt.quadTimesKMS.init(kernels=kernels, t=legQuadPoints)
        EM_JAXopt.spikesTimesKMS = kernelsMatricesStore.IndPointsLocsAndSpikesTimesKMS
        EM_JAXopt.spikesTimesKMS.init(kernels=kernels, t=spikesTimesArray)
        EM_JAXopt.reg_param = reg_param
        EM_JAXopt.eLL = expectedLogLikelihood.PointProcessELLExpLink
        EM_JAXopt.eLL.init(legQuadWeights=legQuadWeights, validSpikesTimesMask=validSpikesTimesMask)

    def maximizeECM(params0, optim_params):
        def variationalParamsOptimFunc(variational_params, all_params):
            value = EM_JAXopt._eval_func(
                vMean=variational_params["variational_mean"],
                vChol=variational_params["variational_chol_vecs"],
                C=all_params["C"], d=all_params["d"],
                kernels_params=all_params["kernels_params"],
                ind_points_locs=all_params["ind_points_locs"],
            )
            return value

        def preIntensityParamsOptimFunc(preIntensity_params, all_params):
            value = EM_JAXopt._eval_func(
                vMean=all_params["variational_mean"],
                vChol=all_params["variational_chol_vecs"],
                C=preIntensity_params["C"], d=preIntensity_params["d"],
                kernels_params=all_params["kernels_params"],
                ind_points_locs=all_params["ind_points_locs"],
            )
            return value

        def kernelsParamsOptimFunc(kernels_params, all_params):
            value = EM_JAXopt._eval_func(
                vMean=all_params["variational_mean"],
                vChol=all_params["variational_chol_vecs"],
                C=all_params["C"], d=all_params["d"],
                kernels_params=kernels_params["kernels_params"],
                ind_points_locs=all_params["ind_points_locs"],
            )
            return value

        def indPointsLocsOptimFunc(indPoints_params, all_params):
            value = EM_JAXopt._eval_func(
                vMean=all_params["variational_mean"],
                vChol=all_params["variational_chol_vecs"],
                C=all_params["C"], d=all_params["d"],
                kernels_params=all_params["kernels_params"],
                ind_points_locs=indPoints_params["ind_points_locs"],
            )
            return value

        params = params0

        variational_solver = jaxopt.LBFGS(fun=variationalParamsOptimFunc,
                                          **optim_params["LBFGS"])
        variational_params = {k: params[k] for k in ('variational_mean', 'variational_chol_vecs')}
        variational_state = variational_solver.init_state(variational_params,
                                                          all_params=params)

        preIntensity_solver = jaxopt.LBFGS(fun=preIntensityParamsOptimFunc,
                                        **optim_params["LBFGS"])
        preIntensity_params = {k: params[k] for k in ('C', 'd')}
        preIntensity_state = preIntensity_solver.init_state(preIntensity_params,
                                                      all_params=params)

        kernels_solver = jaxopt.LBFGS(fun=kernelsParamsOptimFunc,
                                      **optim_params["LBFGS"])
        kernels_params = {"kernels_params": params["kernels_params"]}
        kernels_state = kernels_solver.init_state(kernels_params,
                                                  all_params=params)

        indPointsLocs_solver = jaxopt.LBFGS(fun=indPointsLocsOptimFunc,
                                            **optim_params["LBFGS"])
        indPointsLocs_params = {"ind_points_locs": params["ind_points_locs"]}
        indPointsLocs_state = indPointsLocs_solver.init_state(indPointsLocs_params, all_params=params)

        for i in range(optim_params["n_em_iterations"]):
            if i > 0 and optim_params["estVariationalParams"]:
                for i in range(optim_params["n_variational_iter"]):
                    variational_params, variational_state = variational_solver.update(params=variational_params, state=variational_state, all_params=params)
                params["variational_mean"] = variational_params["variational_mean"]
                params["variational_chol_vecs"] = variational_params["variational_chol_vecs"]
                # print(f"Variational={-variational_state.value}")

            if optim_params["estPreIntensityParams"]:
                for i in range(optim_params["n_preIntensity_iter"]):
                    preIntensity_params, preIntensity_state = preIntensity_solver.update(params=preIntensity_params, state=preIntensity_state, all_params=params)
                params["C"] = preIntensity_params["C"]
                params["d"] = preIntensity_params["d"]
                print(f"PreIntensity={-preIntensity_state.value}")
                breakpoint()

            if optim_params["estKernelsParams"]:
                for i in range(optim_params["n_kernels_iter"]):
                    kernels_params, kernels_state = kernels_solver.update(params=kernels_params, state=kernels_state, all_params=params)
                params["kernels_params"] = kernels_params["kernels_params"]
                # print(f"Kernels={-kernels_state.value}")

            if optim_params["estIndPointsParams"]:
                for i in range(optim_params["n_indPointsLocs_iter"]):
                    indPointsLocs_params, indPointsLocs_state = indPointsLocs_solver.update(params=indPointsLocs_params, state=indPointsLocs_state, all_params=params)
                params["ind_points_locs"] = indPointsLocs_params["ind_points_locs"]
                # print(f"IndPointsLocs={-indPointsLocs_state.value}")

        return params

    def maximize(params0, optim_params):
        solver = jaxopt.LBFGS(fun=EM_JAXopt._eval_func_params_as_list, **optim_params)
        res = solver.run(params0)
        return res

    def maximizeInSteps(params0, optim_params):
        solver = jaxopt.LBFGS(fun=EM_JAXopt._eval_func_params_as_list, **optim_params)
        print("About to call solver.init_state(params)")
        state = solver.init_state(params0)
        print("Called solver.init_state(params) done")

        params = params0
        for step in range(optim_params["maxiter"]):
            params, state = solver.update(params=params, state=state)
            lower_bound = -state.value
            print(f"Iteration {step}: {lower_bound}")
        return params, state

    def maximize_jaxopt_scipy(params0, optim_params):
        def mycallback(params):
            lb = -1*EM_JAXopt._eval_func_params_as_list(params)
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
        answer = -1*svlb.eval(vMean=vMean, vCov=vCov, C=C, d=d, Kzz=Kzz,
                              Kzz_cho=Kzz_cho, KtzQuad=Ktz_quad,
                              KtzSpikes=Ktz_spikes, KttDiag=1.0)
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
