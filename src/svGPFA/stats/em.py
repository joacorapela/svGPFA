
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

class EM_JAX:

    def init(spikesTimesArray, validSpikesTimesMask, kernels, legQuadPoints,
             legQuadWeights, reg_param):
        EM_JAX.indPointsLocsKMS = kernelsMatricesStore.IndPointsLocsKMS_Chol
        EM_JAX.indPointsLocsKMS.init(kernels=kernels)
        EM_JAX.quadTimesKMS = kernelsMatricesStore.IndPointsLocsAndQuadTimesKMS
        EM_JAX.quadTimesKMS.init(kernels=kernels, t=legQuadPoints)
        EM_JAX.spikesTimesKMS = kernelsMatricesStore.IndPointsLocsAndSpikesTimesKMS
        EM_JAX.spikesTimesKMS.init(kernels=kernels, t=spikesTimesArray)
        EM_JAX.reg_param = reg_param
        EM_JAX.eLL = expectedLogLikelihood.PointProcessELLExpLink
        EM_JAX.eLL.init(legQuadWeights=legQuadWeights, validSpikesTimesMask=validSpikesTimesMask)

    def maximizeECM(params0, optim_params, n_em_iterations=10):
        def getVariationalParamsOptimFunc(params):
            def optimFunc(variational_params):
                params["variational_mean"] = \
                    variational_params["variational_mean"]
                params["variational_chol_vecs"] = \
                    variational_params["variational_chol_vecs"]
                value = EM_JAX._eval_func(params)
                return value
            return optimFunc

        params = params0
        for i in range(n_em_iterations):
            # variational params
            variational_params = {k: params[k] for k in ('variational_mean', 'variational_chol_vecs')}
            variational_params_optim_func = getVariationalParamsOptimFunc(params)
            solver = jaxopt.LBFGS(fun=variational_params_optim_func, **optim_params)
            res = solver.run(variational_params)
            breakpoint()
            params["variational_mean"] = res.params["variational_mean"]
            params["variational_chol_vecs"] = res.params["variational_chol_vecs"]
        return params

    def maximize(params0, optim_params):
#         solver = jaxopt.ScipyMinimize(fun=self._eval_func,
#                                       method="L-BFGS-B",
#                                       callback=mycallback,
#                                       **optim_params)
#         def mycallback(params):
#             lb = -1*self._eval_func(params)
#             print(f"lower bound: {lb}")

        eval0 = EM_JAX._eval_func(params=params0)
        assert(math.isfinite(eval0))

        solver = jaxopt.LBFGS(fun=EM_JAX._eval_func, **optim_params)
        res = solver.run(params0)
        return res

#     def maximize_scipy(self, params0, optim_params):
#         print("Jitting optim functions")
#         eval_func_jitted = jax.jit(self._eval_func)
#         eval_func_grad_jitted = jax.jit(jax.grad(self._eval_func))
#         print("Jitting optim functions done")
# 
#         print("Evaluating optimisation function once")
#         eval0 = eval_func_jitted(params=params0)
#         assert(math.isfinite(eval0))
#         print(f"Evaluating optimisation function done with value {eval0}")
# 
#         def mycallback(params):
#             lb = -1*self._eval_func(params)
#             print(f"lower bound: {lb}")
# 
#         print("Starting minimization")
#         res = scipy.optimize.minimize(fun=eval_func_jitted,
#                                       x0=params0,
#                                       jac=eval_func_grad_jitted,
#                                       method="L-BFGS-B",
#                                       callback=mycallback,
#                                       options=optim_params)
#         print("minimization done")
#         return res

    def maximize_jaxopt_scipy(params0, optim_params):
        eval_func_jitted = jax.jit(EM_JAX._eval_func)
        eval0 = eval_func_jitted(params=params0)
        assert(math.isfinite(eval0))

        def mycallback(params):
            lb = -1*EM_JAX._eval_func(params)
            print(f"lower bound: {lb}")

        solver = jaxopt.ScipyMinimize(fun=eval_func_jitted,
                                      method="L-BFGS-B",
                                      callback=mycallback,
                                      **optim_params)
        # solver = jaxopt.LBFGS(fun=self._eval_func, **optim_params)
        res = solver.run(params0)
        breakpoint()
        return res

    def maximizeInSteps(params0, optim_params):
        eval_func_jitted = jax.jit(EM_JAX._eval_func)
        solver = jaxopt.LBFGS(fun=eval_func_jitted, **optim_params)
        # solver = jaxopt.LBFGS(fun=self._eval_func, **optim_params)
        params = params0
        print("About to call solver.init_state(params)")
        state = solver.init_state(params)
        print("Called solver.init_state(params) done")

        for step in range(optim_params["maxiter"]):
            params, state = solver.update(params=params, state=state)
            lower_bound = -state.value
            print(f"Iteration {step}: {lower_bound}")
        return state

    def _eval_func(params):
        vMean = params["variational_mean"]
        vChol = params["variational_chol_vecs"]
        C = params["C"]
        d = params["d"]
        kernels_params = params["kernels_params"]
        ind_points_locs = params["ind_points_locs"]

        Kzz, Kzz_inv = EM_JAX.indPointsLocsKMS.buildKernelsMatrices(
            kernels_params=kernels_params, ind_points_locs=ind_points_locs,
            reg_param=EM_JAX.reg_param)
        Ktz_quad = EM_JAX.quadTimesKMS.buildKernelsMatrices(
            kernels_params=kernels_params, ind_points_locs=ind_points_locs)
        Ktz_spikes = EM_JAX.spikesTimesKMS.buildKernelsMatrices(
            kernels_params=kernels_params, ind_points_locs=ind_points_locs)
        vCov = buildCovsFromCholVecs(vChol)
        svlb = svLowerBound.SVLowerBound
        answer = -1*svlb.eval(vMean=vMean, vCov=vCov, C=C, d=d, Kzz=Kzz,
                              Kzz_inv=Kzz_inv, KtzQuad=Ktz_quad,
                              KtzSpikes=Ktz_spikes, KttDiag=1.0)
        return answer

class EM(abc.ABC):

    @abc.abstractmethod
    def maximizeInSteps(self, model, optim_params, method="ECM", getIterationModelParamsFn=None,
                 printIterationModelParams=True,
                 logLock=None, logStreamFN=None,
                 lowerBoundLock=None, lowerBoundStreamFN=None,
                 latentsTimes=None, latentsLock=None, latentsStreamFN=None,
                 verbose=True, out=sys.stdout,
                 savePartial=False,
                 savePartialFilenamePattern="results/00000000_{:s}_estimatedModel.pickle",
                ):
        """Maximizes the sparse variational lower bound, Eq. 4 in Duncker and
        Sahani, 2018.

        .. note::
            Only parameters **model**, **optim_params** and **method** should
            be set by users. The remaining parameters are used to interface
            :meth:`svGPFA.stats.svEM.EM.maximize` with the dashboard.

        :param model: svGPFA model used to calculate the lower bound
        :type  model: :class:`svGPFA.stats.svLowerBound.SVLowerBound`
        :param optim_params: optimization parameters. The format of this dictionary is identical to the one described in :any:`optim_params`. It can be obtained at the key ``optim_params`` of the dictionary returned by :func:`svGPFA.utils.initUtils.getParamsAndKernelsTypes`.
        :type  optim_params: dictionary
        :param method: either ECM for Expectation Conditional Maximization or mECM for multicycle expectation conditional maximization. Refer to :cite:t:`mcLachlanAndKrishnan08` for details on these algorithms.
        """
        pass


    @abc.abstractmethod
    def _eStep(self, model, optim_params):
        pass


    @abc.abstractmethod
    def _mStepPreIntensity(self, model, optim_params):
        pass


    @abc.abstractmethod
    def _mStepKernels(self, model, optim_params):
        pass


    @abc.abstractmethod
    def _mStepIndPointsLocs(self, model, optim_params):
        pass


    def _writeToLockedLog(self, message, logLock, logStream, logStreamFN):
        logStream.write(message)
        if logLock is not None and not logLock.is_locked():
            logLock.lock()
            with open(logStreamFN, 'a') as f:
                f.write(logStream.getvalue())
            logLock.unlock()
            logStream.truncate(0)
            logStream.seek(0)


class EM_PyTorch(EM):

    def maximizeSimultaneously(self, model, optim_params, getIterationModelParamsFn=None,
                               printIterationModelParams=True,
                               logLock=None, logStreamFN=None,
                               lowerBoundLock=None, lowerBoundStreamFN=None,
                               latentsTimes=None, latentsLock=None, latentsStreamFN=None,
                               verbose=True, out=sys.stdout,
                               savePartial=False,
                               savePartialFilenamePattern="results/00000000_{:s}_estimatedModel.pickle",
                              ):
        if latentsStreamFN is not None and latentsTimes is None:
            raise RuntimeError("Please specify latentsTime if you want to save latents")

        iter = 0
        if savePartial:
            savePartialFilename = savePartialFilenamePattern.format("initial")
            resultsToSave = {"model": model}
            with open(savePartialFilename, "wb") as f: pickle.dump(resultsToSave, f)
        lowerBound0 = model.eval()
        lowerBoundHist = [lowerBound0.item()]
        elapsedTimeHist = [0.0]
        startTime = time.time()

        if lowerBoundLock is not None and lowerBoundStreamFN is not None and not lowerBoundLock.is_locked():
            lowerBoundLock.lock()
            with open(lowerBoundStreamFN, 'wb') as f:
                np.save(f, np.array(lowerBoundHist))
            lowerBoundLock.unlock()

        if latentsLock is not None and latentsStreamFN is not None and not latentsLock.is_locked():
            latentsLock.lock()
            muK, varK = model.predictLatents(newTimes=latentsTimes)

            with open(latentsStreamFN, 'wb') as f:
                np.savez(f, iteration=iter, times=latentsTimes.detach().numpy(), muK=muK.detach().numpy(), varK=varK.detach().numpy())
            lowerBoundLock.unlock()
        logStream = io.StringIO()
        if getIterationModelParamsFn is not None:
            iterationModelParams = getIterationModelParamsFn(model=model)
            if printIterationModelParams:
                print(iterationModelParams)
            iterationsModelParams = [None for i in
                                     range(optim_params["em_max_iter"]+1)]
            iterationsModelParams[iter] = iterationModelParams
        else:
            iterationsModelParams = None
        maxRes = {"lowerBound": -math.inf}
        iter += 1
        while iter <= optim_params["em_max_iter"]:
            message = "Iteration {:02d}, start: {:f}\n".format(
                iter, maxRes["lowerBound"])
            if verbose:
                out.write(message)
            self._writeToLockedLog(
                message=message,
                logLock=logLock,
                logStream=logStream,
                logStreamFN=logStreamFN
            )
            try:
                maxRes = self._allSteps(
                    model=model,
                    optim_params=optim_params["allsteps_optim_params"])
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(e)
                print(stack_trace)
                terminationInfo = ErrorTerminationInfo(
                    message=f"Error occured in iteration {iter}",
                    error=e, stack_trace=stack_trace)
                return lowerBoundHist, elapsedTimeHist, \
                       terminationInfo, iterationsModelParams
            message = "Iteration {:02d}, end: {:f}, niter: {:d}, nfeval: {:d}\n".format(
                iter, maxRes["lowerBound"], maxRes["niter"],
                maxRes["nfeval"])
            if verbose:
                out.write(message)
                self._writeToLockedLog(
                    message=message,
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                )
                if savePartial:
                    savePartialFilename = \
                        savePartialFilenamePattern.format(
                            "{:s}{:03d}".format("allStpes", iter))
                    resultsToSave = {"model": model}
                    with open(savePartialFilename, "wb") as f:
                        pickle.dump(resultsToSave, f)
                if getIterationModelParamsFn is not None:
                    iterationModelParams = getIterationModelParamsFn(model=model)
                    if printIterationModelParams:
                        print(iterationModelParams)
                    iterationsModelParams[iter] = iterationModelParams
            iter += 1
        elapsedTimeHist.append(time.time()-startTime)
        lowerBoundHist.append(maxRes["lowerBound"].item())

        if lowerBoundLock is not None and \
           lowerBoundStreamFN is not None and \
           not lowerBoundLock.is_locked():
            lowerBoundLock.lock()
            with open(lowerBoundStreamFN, 'wb') as f:
                np.save(f, np.array(lowerBoundHist))
            lowerBoundLock.unlock()

        if latentsLock is not None and \
           latentsStreamFN is not None and \
           not latentsLock.is_locked():
            latentsLock.lock()
            muK, varK = model.predictLatents(newTimes=latentsTimes)

            with open(latentsStreamFN, 'wb') as f:
                np.savez(f, iteration=iter,
                         times=latentsTimes.detach().numpy(),
                         muK=muK.detach().numpy(),
                         varK=varK.detach().numpy())
            lowerBoundLock.unlock()

        iter += 1
        terminationInfo = TerminationInfo(
            "Maximum number of iterations ({:d}) reached".format(
                optim_params["em_max_iter"]))
        return lowerBoundHist, elapsedTimeHist, terminationInfo, \
            iterationsModelParams

    def maximizeInSteps(self, model, optim_params, method="ECM", getIterationModelParamsFn=None,
                        printIterationModelParams=True,
                        logLock=None, logStreamFN=None,
                        lowerBoundLock=None, lowerBoundStreamFN=None,
                        latentsTimes=None, latentsLock=None, latentsStreamFN=None,
                        verbose=True, out=sys.stdout,
                        savePartial=False,
                        savePartialFilenamePattern="results/00000000_{:s}_estimatedModel.pickle"):
        if latentsStreamFN is not None and latentsTimes is None:
            raise RuntimeError("Please specify latentsTime if you want to save latents")

        iter = 0
        if savePartial:
            savePartialFilename = savePartialFilenamePattern.format("initial")
            resultsToSave = {"model": model}
            with open(savePartialFilename, "wb") as f: pickle.dump(resultsToSave, f)
        lowerBound0 = model.eval()
        lowerBoundHist = [lowerBound0.item()]
        elapsedTimeHist = [0.0]
        startTime = time.time()

        if lowerBoundLock is not None and lowerBoundStreamFN is not None and not lowerBoundLock.is_locked():
            lowerBoundLock.lock()
            with open(lowerBoundStreamFN, 'wb') as f:
                np.save(f, np.array(lowerBoundHist))
            lowerBoundLock.unlock()

        if latentsLock is not None and latentsStreamFN is not None and not latentsLock.is_locked():
            latentsLock.lock()
            muK, varK = model.predictLatents(newTimes=latentsTimes)

            with open(latentsStreamFN, 'wb') as f:
                np.savez(f, iteration=iter, times=latentsTimes.detach().numpy(), muK=muK.detach().numpy(), varK=varK.detach().numpy())
            lowerBoundLock.unlock()
        logStream = io.StringIO()
        if method.lower() == "ecm":
            steps = ["estep", "mstep_embedding", "mstep_kernels",
                     "mstep_indpointslocs"]
        elif method.lower() == "mecm":
            # see McLachlan, G. J., & Krishnan, T. (2007). The EM algorithm and
            # extensions (Vol. 382). John Wiley & Sons) Chapter 5
            steps = ["estep", "mstep_embedding", "estep", "mstep_kernels",
                     "estep", "mstep_indpointslocs"]
        else:
            raise ValueError("Invalid method={:s}. Supported values are ECM and mECM".format(method))
        functions_for_steps = {"estep": self._eStep,
                               "mstep_embedding": self._mStepPreIntensity,
                               "mstep_kernels": self._mStepKernels,
                               "mstep_indpointslocs": self._mStepIndPointsLocs}
        if getIterationModelParamsFn is not None:
            iterationModelParams = getIterationModelParamsFn(model=model)
            if printIterationModelParams:
                print(iterationModelParams)
            iterationsModelParams = [None for i in
                                     range(optim_params["em_max_iter"]+1)]
            iterationsModelParams[iter] = iterationModelParams
        else:
            iterationsModelParams = None
        maxRes = {"lowerBound": -math.inf}
        iter += 1
        while iter <= optim_params["em_max_iter"]:
            for step in steps:
                if optim_params["{:s}_estimate".format(step)]:
                    message = "Iteration {:02d}, {:s} start: {:f}\n".format(
                        iter, step, maxRes["lowerBound"])
                    if verbose:
                        out.write(message)
                    self._writeToLockedLog(
                        message=message,
                        logLock=logLock,
                        logStream=logStream,
                        logStreamFN=logStreamFN
                    )
                    try:
                        maxRes = functions_for_steps[step](
                            model=model,
                            optim_params=optim_params["{:s}_optim_params".format(
                                step)])
                    except Exception as e:
                        stack_trace = traceback.format_exc()
                        print(e)
                        print(stack_trace)
                        terminationInfo = ErrorTerminationInfo(
                            message=f"Error occured while processing {step} in iteration {iter}",
                            error=e, stack_trace=stack_trace)
                        return lowerBoundHist, elapsedTimeHist, \
                               terminationInfo, iterationsModelParams
                    message = "Iteration {:02d}, {:s} end: {:f}, niter: {:d}, nfeval: {:d}\n".format(
                        iter, step, maxRes["lowerBound"], maxRes["niter"],
                        maxRes["nfeval"])
                    if verbose:
                        out.write(message)
                    self._writeToLockedLog(
                        message=message,
                        logLock=logLock,
                        logStream=logStream,
                        logStreamFN=logStreamFN
                    )
                    if savePartial:
                        savePartialFilename = \
                            savePartialFilenamePattern.format(
                                "{:s}{:03d}".format(step, iter))
                        resultsToSave = {"model": model}
                        with open(savePartialFilename, "wb") as f:
                            pickle.dump(resultsToSave, f)
                    if getIterationModelParamsFn is not None:
                        iterationModelParams = getIterationModelParamsFn(model=model)
                        if printIterationModelParams:
                            print(iterationModelParams)
                        iterationsModelParams[iter] = iterationModelParams
            elapsedTimeHist.append(time.time()-startTime)
            lowerBoundHist.append(maxRes["lowerBound"].item())

            if lowerBoundLock is not None and \
               lowerBoundStreamFN is not None and \
               not lowerBoundLock.is_locked():
                lowerBoundLock.lock()
                with open(lowerBoundStreamFN, 'wb') as f:
                    np.save(f, np.array(lowerBoundHist))
                lowerBoundLock.unlock()

            if latentsLock is not None and \
               latentsStreamFN is not None and \
               not latentsLock.is_locked():
                latentsLock.lock()
                muK, varK = model.predictLatents(newTimes=latentsTimes)

                with open(latentsStreamFN, 'wb') as f:
                    np.savez(f, iteration=iter,
                             times=latentsTimes.detach().numpy(),
                             muK=muK.detach().numpy(),
                             varK=varK.detach().numpy())
                lowerBoundLock.unlock()

            iter += 1
        terminationInfo = TerminationInfo(
            "Maximum number of iterations ({:d}) reached".format(
                optim_params["em_max_iter"]))
        return lowerBoundHist, elapsedTimeHist, terminationInfo, \
            iterationsModelParams

    def _allSteps(self, model, optim_params):
        x = []
        x.extend(model.getVariationalDistParams())
        x.extend(model.getPreIntensityParams())
        x.extend(model.getKernelsParams())
        x.extend(model.getIndPointsLocs())
        def evalFunc():
            model.buildKernelsMatrices()
            model.buildVariationalCov()
            answer = model.eval()
            # begin debug
            # print(model.getVariationalDistParams()[2][0,:,0])
            # end debug
            return answer
        optimizer = torch.optim.LBFGS(x, **optim_params)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer)
        return answer

    def _eStep(self, model, optim_params):
        x = model.getVariationalDistParams()
        def evalFunc():
            model.buildVariationalCov()
            answer = model.eval()
            return answer
        optimizer = torch.optim.LBFGS(x, **optim_params)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer)
        return answer

    def _mStepPreIntensity(self, model, optim_params):
        x = model.getPreIntensityParams()
        posteriorOnLatentsStats = model.computePosteriorOnLatentsStats()
        evalFunc = lambda: model.evalELLSumAcrossTrialsAndNeurons(posteriorOnLatentsStats=posteriorOnLatentsStats)
        # evalFunc = model.eval
        optimizer = torch.optim.LBFGS(x, **optim_params)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer)
        return answer

    def _mStepKernels(self, model, optim_params):
        x = model.getKernelsParams()
        prev_x = [copy.deepcopy(anx) for anx in x]
        def evalFunc():
            model.buildKernelsMatrices()
            answer = model.eval()
            return answer
        optimizer = torch.optim.LBFGS(x, **optim_params)
        try:
            answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer)
            # begin debug
            # print("*** Updated kernels params ***")
            # print(x)
            # raise ValueError("Test error in _mStepKernels")
            # end debug
        except Exception:
            x = [anx.detach() for anx in x]
            for i in range(len(x)):
                x[i][:] = prev_x[i][:]
            raise
        return answer

    def _mStepIndPointsLocs(self, model, optim_params):
        x = model.getIndPointsLocs()
        def evalFunc():
            model.buildKernelsMatrices()
            answer = model.eval()
            return answer
        optimizer = torch.optim.LBFGS(x, **optim_params)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer)
        return answer

    def _setupAndMaximizeStep(self, x, evalFunc, optimizer):
        for i in range(len(x)):
            x[i].requires_grad = True
        maxRes = self._maximizeStep(evalFunc=evalFunc, optimizer=optimizer)
        for i in range(len(x)):
            x[i].requires_grad = False
        return maxRes

    def _maximizeStep(self, evalFunc, optimizer):
        def closure():
            optimizer.zero_grad()
            curEval = -evalFunc()
            curEval.backward(retain_graph=True)
            return curEval
        optimizer.step(closure)
        lowerBound = evalFunc()
        stateOneEpoch = optimizer.state[optimizer._params[0]]
        nfeval = stateOneEpoch["func_evals"]
        niter = stateOneEpoch["n_iter"]
        return {"lowerBound": lowerBound, "nfeval": nfeval, "niter": niter}

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
