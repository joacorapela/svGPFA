
import pdb
import sys
import io
import time
import math
import pickle
import numpy as np
import scipy.optimize
import torch

class SVEM:

    def maximize(self, model, optimParams,
                 logLock=None, logStreamFN=None,
                 lowerBoundLock=None, lowerBoundStreamFN=None,
                 latentsTimes=None, latentsLock=None, latentsStreamFN=None,
                 verbose=True, out=sys.stdout,
                 savePartial=False,
                 savePartialFilenamePattern="00000000_{:s}_estimatedModel.pickle",
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
        iter += 1
        logStream = io.StringIO()
        steps = ["estep", "mstep_embedding", "mstep_kernels", "mstep_indpointslocs"]
        functions_for_steps = {
            "estep": self._eStep,
            "mstep_embedding": self._mStepEmbedding,
            "mstep_kernels": self._mStepKernels,
            "mstep_indpointslocs": self._mStepIndPointsLocs,
        }
        maxRes = {"maximum": -math.inf}
        while iter<optimParams["em_max_iter"]:
            for step in steps:
                if optimParams["{:s}_estimate".format(step)]:
                    message = "Iteration {:02d}, {:s} start: {:f}\n".format(iter, step, maxRes["maximum"])
                    if verbose:
                        out.write(message)
                    self._writeToLockedLog(
                        message=message,
                        logLock=logLock,
                        logStream=logStream,
                        logStreamFN=logStreamFN
                    )
                    maxRes = functions_for_steps[step](model=model, optimParams=optimParams["{:s}_optim_params".format(step)])
                    message = "Iteration {:02d}, {:s} end: {:f}, niter: {:d}, nfeval: {:d}\n".format(iter, step, maxRes["maximum"], maxRes["niter"], maxRes["nfeval"])
                    if verbose:
                        out.write(message)
                    self._writeToLockedLog(
                        message=message,
                        logLock=logLock,
                        logStream=logStream,
                        logStreamFN=logStreamFN
                    )
                    if savePartial:
                        savePartialFilename = savePartialFilenamePattern.format("{:s}{:03d}".format(step, iter))
                        resultsToSave = {"model": model}
                        with open(savePartialFilename, "wb") as f: pickle.dump(resultsToSave, f)
            elapsedTimeHist.append(time.time()-startTime)
            lowerBoundHist.append(maxRes["maximum"])

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

            iter += 1
        return lowerBoundHist, elapsedTimeHist

    def _eStep(self, model, optimParams, method="L-BFGS-B"):
        def eval_func(z):
            model.set_svPosteriorOnIndPoints_params_from_flattened(flattened_params=z.tolist())
            model.set_svPosteriorOnIndPoints_params_requires_grad(requires_grad=True)
            value = -model.eval()
            value.backward(retain_graph=True)
            grad_list = model.get_flattened_svPosteriorOnIndPoints_params_grad()
            value = value.item()
            grad = np.array(grad_list)
            return (value, grad)

        z0 = np.array(model.get_flattened_svPosteriorOnIndPoints_params())
        optim_res = scipy.optimize.minimize(fun=eval_func, x0=z0,
                                            method=method, jac=True,
                                            options=optimParams)
        model.set_svPosteriorOnIndPoints_params_requires_grad(requires_grad=False)
        model.set_svPosteriorOnIndPoints_params_from_flattened(flattened_params=optim_res.x.tolist())
        answer = {"maximum": -optim_res.fun, "niter": optim_res.nit, "nfeval": optim_res.nfev}
        return answer

    def _mStepEmbedding(self, model, optimParams, method="L-BFGS-B"):
        def eval_func(z):
            model.set_svEmbedding_params_from_flattened(flattened_params=z.tolist())
            model.set_svEmbedding_params_requires_grad(requires_grad=True)
#             svPosteriorOnLatentsStats = model.computeSVPosteriorOnLatentsStats()
#             value = -model.evalELLSumAcrossTrialsAndNeurons(svPosteriorOnLatentsStats=svPosteriorOnLatentsStats)
            value = -model.eval()
            value.backward(retain_graph=True)
            grad_list = model.get_flattened_svEmbedding_params_grad()
            value = value.item()
            grad = np.array(grad_list)
            return (value, grad)

        z0 = np.array(model.get_flattened_svEmbedding_params())
        optim_res = scipy.optimize.minimize(fun=eval_func, x0=z0,
                                            method=method, jac=True,
                                            options=optimParams)
        model.set_svEmbedding_params_requires_grad(requires_grad=False)
        model.set_svEmbedding_params_from_flattened(flattened_params=optim_res.x.tolist())
        answer = {"maximum": -optim_res.fun, "niter": optim_res.nit, "nfeval": optim_res.nfev}
        return answer

    def _mStepKernels(self, model, optimParams, method="L-BFGS-B"):
        def eval_func(z):
            model.set_kernels_params_from_flattened(flattened_params=z.tolist())
            model.set_kernels_params_requires_grad(requires_grad=True)
            model.buildKernelsMatrices()
            value = -model.eval()
            value.backward(retain_graph=True)
            grad_list = model.get_flattened_kernels_params_grad()
            value = value.item()
            grad = np.array(grad_list)
            return (value, grad)

        z0 = np.array(model.get_flattened_kernels_params())
        optim_res = scipy.optimize.minimize(fun=eval_func, x0=z0,
                                            method=method, jac=True,
                                            options=optimParams)
        model.set_kernels_params_requires_grad(requires_grad=False)
        model.set_kernels_params_from_flattened(flattened_params=optim_res.x.tolist())
        answer = {"maximum": -optim_res.fun, "niter": optim_res.nit, "nfeval": optim_res.nfev}
        print("*** kernels parameters: {}".format(optim_res.x))
        return answer

    def _mStepIndPointsLocs(self, model, optimParams, method="L-BFGS-B"):
        def eval_func(z):
            model.set_indPointsLocs_from_flattened(flattened_params=z.tolist())
            model.set_indPointsLocs_requires_grad(requires_grad=True)
            model.buildKernelsMatrices()
            value = -model.eval()
            value.backward(retain_graph=True)
            grad_list = model.get_flattened_indPointsLocs_grad()
            value = value.item()
            grad = np.array(grad_list)
            return (value, grad)

        z0 = np.array(model.get_flattened_indPointsLocs())
        optim_res = scipy.optimize.minimize(fun=eval_func, x0=z0,
                                            method=method, jac=True,
                                            options=optimParams)
        model.set_indPointsLocs_requires_grad(requires_grad=False)
        model.set_indPointsLocs_from_flattened(flattened_params=optim_res.x.tolist())
        answer = {"maximum": -optim_res.fun, "niter": optim_res.nit, "nfeval": optim_res.nfev}
        return answer

    def _writeToLockedLog(self, message, logLock, logStream, logStreamFN):
        logStream.write(message)
        if logLock is not None and not logLock.is_locked():
            logLock.lock()
            with open(logStreamFN, 'a') as f:
                f.write(logStream.getvalue())
            logLock.unlock()
            logStream.truncate(0)
            logStream.seek(0)

