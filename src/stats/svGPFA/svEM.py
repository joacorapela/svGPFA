
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

    def maximize(self, model, optimParams, method="EM",
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
        if method=="EM":
            candidate_steps = ["estep", "mstep_embedding", "mstep_kernels", "mstep_indpointslocs"]
            steps = []
            for candidate_step in candidate_steps:
                if optimParams["{:s}_estimate".format(candidate_step)]:
                    steps.append(candidate_step)
        elif method=="mECM":
            candidate_steps = ["mstep_embedding", "mstep_kernels", "mstep_indpointslocs"]
            steps = []
            for candidate_step in candidate_steps:
                if optimParams["{:s}_estimate".format(candidate_step)]:
                    steps.append("estep")
                    steps.append(candidate_step)
        else:
            raise ValueError("Invalid method=={:s}".format(method))
        functions_for_steps = {
            "estep": self._eStep,
            "mstep_embedding": self._mStepEmbedding,
            "mstep_kernels": self._mStepKernels,
            "mstep_indpointslocs": self._mStepIndPointsLocs,
        }
        maxRes = {"lowerBound": -math.inf}
        while iter<optimParams["em_max_iter"]:
            for step in steps:
                if optimParams["{:s}_estimate".format(step)]:
                    message = "Iteration {:02d}, {:s} start: {:f}\n".format(iter, step, maxRes["lowerBound"])
                    if verbose:
                        out.write(message)
                    self._writeToLockedLog(
                        message=message,
                        logLock=logLock,
                        logStream=logStream,
                        logStreamFN=logStreamFN
                    )
                    maxRes = functions_for_steps[step](model=model, optimParams=optimParams["{:s}_optim_params".format(step)])
                    message = "Iteration {:02d}, {:s} end: {:f}, niter: {:d}, nfeval: {:d}\n".format(iter, step, maxRes["lowerBound"], maxRes["niter"], maxRes["nfeval"])
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
            lowerBoundHist.append(maxRes["lowerBound"])

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
        def eval_func(x_torch_flat):
            model.set_svPosteriorOnIndPoints_params_from_flattened(flattened_params=x_torch_flat)
            value = -model.eval()
            return value

        x0 = model.get_flattened_svPosteriorOnIndPoints_params().detach().numpy()
        fun = lambda x_numpy_flat: self._eval_func_wrapper(x_numpy_flat=x_numpy_flat, eval_func=eval_func)
        hessp = lambda x, p: self._hessian_prod(x, p, eval_func=eval_func)
        optim_res = scipy.optimize.minimize(fun=fun, x0=x0, method=method, jac=True, hessp=hessp, options=optimParams)
        model.set_svPosteriorOnIndPoints_params_from_flattened(flattened_params=torch.from_numpy(optim_res.x))
        answer = {"lowerBound": -optim_res.fun, "niter": optim_res.nit, "nfeval": optim_res.nfev}
        print("*** variational parameters: {}".format(optim_res.x))
        # pdb.set_trace()
        return answer

    def _mStepEmbedding(self, model, optimParams, method="L-BFGS-B"):
        def eval_func(x_torch_flat):
            model.set_svEmbedding_params_from_flattened(flattened_params=x_torch_flat)
            svPosteriorOnLatentsStats = model.computeSVPosteriorOnLatentsStats()
            value = -model.evalELLSumAcrossTrialsAndNeurons(svPosteriorOnLatentsStats=svPosteriorOnLatentsStats)
            # value = -model.eval()
            return value

        x0 = model.get_flattened_svEmbedding_params().detach().numpy()
        fun = lambda x_numpy_flat: self._eval_func_wrapper(x_numpy_flat=x_numpy_flat, eval_func=eval_func)
        hessp = lambda x, p: self._hessian_prod(x, p, eval_func=eval_func)
        C = model.getSVEmbeddingParams()[0]
        bounds = [(0, None)]*C.shape[1] + [(None,None)]*(len(x0)-C.shape[1])
        optim_res = scipy.optimize.minimize(fun=fun, x0=x0, method=method, jac=True, hessp=hessp, bounds=bounds, options=optimParams)
        # model.set_svEmbedding_params_requires_grad(requires_grad=False)
        model.set_svEmbedding_params_from_flattened(flattened_params=torch.from_numpy(optim_res.x))
        answer = {"lowerBound": -optim_res.fun, "niter": optim_res.nit, "nfeval": optim_res.nfev}
        print("*** embedding parameters: {}".format(optim_res.x))
        # pdb.set_trace()
        return answer

    def _mStepKernels(self, model, optimParams, method="L-BFGS-B"):

        def eval_func(x_torch_flat):
            model.set_kernels_params_from_flattened(flattened_params=x_torch_flat)
            model.buildKernelsMatrices()
            value = -model.eval()
            # pdb.set_trace()
            return value

        x0 = model.get_flattened_kernels_params().numpy()
        fun = lambda x_numpy_flat: self._eval_func_wrapper(x_numpy_flat=x_numpy_flat, eval_func=eval_func)
        hessp = lambda x, p: self._hessian_prod(x, p, eval_func=eval_func)
        optim_res = scipy.optimize.minimize(fun=fun, x0=x0, method=method, jac=True, hessp=hessp, options=optimParams)
        # model.set_kernels_params_requires_grad(requires_grad=False)
        model.set_kernels_params_from_flattened(flattened_params=torch.from_numpy(optim_res.x))
        answer = {"lowerBound": -optim_res.fun, "niter": optim_res.nit, "nfeval": optim_res.nfev}
        print("*** kernels parameters: {}".format(optim_res.x))
        # pdb.set_trace()
        return answer

    def _mStepIndPointsLocs(self, model, optimParams, method="L-BFGS-B"):

        def eval_func(x_torch_flat):
            model.set_indPointsLocs_from_flattened(flattened_params=x_torch_flat)
            model.buildKernelsMatrices()
            value = -model.eval()
            return value

        x0 = model.get_flattened_indPointsLocs().numpy()
        fun = lambda x_numpy_flat: self._eval_func_wrapper(x_numpy_flat=x_numpy_flat, eval_func=eval_func)
        hessp = lambda x, p: self._hessian_prod(x, p, eval_func=eval_func)
        optim_res = scipy.optimize.minimize(fun=fun, x0=x0, method=method, jac=True, hessp=hessp, options=optimParams)
        # model.set_indpointslocs_params_requires_grad(requires_grad=False)
        model.set_indPointsLocs_from_flattened(flattened_params=torch.from_numpy(optim_res.x))
        answer = {"lowerBound": -optim_res.fun, "niter": optim_res.nit, "nfeval": optim_res.nfev}
        print("*** ind points locs: {}".format(optim_res.x))
        # pdb.set_trace()
        return answer

    def _eval_func_wrapper(self, x_numpy_flat, eval_func):
        x_torch_flat = torch.from_numpy(x_numpy_flat)
        x_torch_flat.requires_grad = True
        value = eval_func(x_torch_flat)
        # value.backward(retain_graph=True)
        value.backward(retain_graph=True)
        x_torch_flat.requires_grad = False
        grad_flat = x_torch_flat.grad
        value = value.item()
        grad = grad_flat.numpy()
        # pdb.set_trace()
        return (value, grad)

    def _hessian_prod(self, x, p, eval_func):
        hp = torch.autograd.functional.hvp(func=eval_func, inputs=torch.from_numpy(x), v=torch.from_numpy(p))[1].detach().numpy()
        # pdb.set_trace()
        return hp

    def _writeToLockedLog(self, message, logLock, logStream, logStreamFN):
        logStream.write(message)
        if logLock is not None and not logLock.is_locked():
            logLock.lock()
            with open(logStreamFN, 'a') as f:
                f.write(logStream.getvalue())
            logLock.unlock()
            logStream.truncate(0)
            logStream.seek(0)

