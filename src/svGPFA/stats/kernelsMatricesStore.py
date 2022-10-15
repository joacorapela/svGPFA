
import pdb
import torch
import abc
import svGPFA.utils.miscUtils


class KernelsMatricesStore(abc.ABC):

    @abc.abstractmethod
    def buildKernelsMatrices(self):
        pass

    def setKernels(self, kernels):
        self._kernels = kernels

    def setInitialParams(self, initial_params):
        self.setIndPointsLocs(
            ind_points_locs=initial_params["inducing_points_locs0"])
        self.setKernelsParams(kernels_params=initial_params["kernels_params0"])

    def setKernelsParams(self, kernels_params):
        for k in range(len(self._kernels)):
            self._kernels[k].setParams(kernels_params[k])

    def setIndPointsLocs(self, ind_points_locs):
        self._ind_points_locs = ind_points_locs

    def getIndPointsLocs(self):
        return self._ind_points_locs

    def getKernels(self):
        return self._kernels

    def getKernelsParams(self):
        answer = []
        for i in range(len(self._kernels)):
            answer.append(self._kernels[i].getParams())
        return answer


class KernelMatricesStoreGettersAndSetters(abc.ABC):
    def get_flattened_kernels_params(self):
        flattened_params = []
        for k in range(len(self._kernels)):
            flattened_params.extend(self._kernels[k].getParams().flatten().tolist())
        return flattened_params

    def get_flattened_kernels_params_grad(self):
        flattened_params_grad = []
        for k in range(len(self._kernels)):
            flattened_params_grad.extend(self._kernels[k].getParams().grad.flatten().tolist())
        return flattened_params_grad

    def set_kernels_params_from_flattened(self, flattened_params):
        for k in range(len(self._kernels)):
            kernel_nParams = self._kernels[k].getParams().numel()
            flattened_param = flattened_params[:kernel_nParams]
            self._kernels[k].setParams(torch.tensor(flattened_param, dtype=torch.double))
            flattened_params = flattened_params[kernel_nParams:]

    def set_kernels_params_requires_grad(self, requires_grad):
        for k in range(len(self._kernels)):
            self._kernels[k].getParams().requires_grad = requires_grad

    def get_flattened_ind_points_locs(self):
        flattened_params = []
        for k in range(len(self._ind_points_locs)):
            flattened_params.extend(self._ind_points_locs[k].flatten().tolist())
        return flattened_params

    def get_flattened_ind_points_locs_grad(self):
        flattened_params_grad = []
        for k in range(len(self._ind_points_locs)):
            flattened_params_grad.extend(self._ind_points_locs[k].grad.flatten().tolist())
        return flattened_params_grad

    def set_ind_points_locs_from_flattened(self, flattened_params):
        for k in range(len(self._ind_points_locs)):
            numel = self._ind_points_locs[k].numel()
            self._ind_points_locs[k] = torch.tensor(flattened_params[:numel],
                                                    dtype=torch.double).reshape(self._ind_points_locs[k].shape)
            flattened_params = flattened_params[numel:]

    def set_ind_points_locs_requires_grad(self, requires_grad):
        for k in range(len(self._ind_points_locs)):
            self._ind_points_locs[k].requires_grad = requires_grad


class IndPointsLocsKMS(KernelsMatricesStore):

    def setRegParam(self, reg_param):
        self._reg_param = reg_param

    # @abc.abstractmethod
    def _invertKzz3D(self, Kzz):
        pass

    @abc.abstractmethod
    def solveForLatent(self, input, latentIndex):
        pass

    @abc.abstractmethod
    def solveForLatentAndTrial(self, input, latentIndex, trialIndex):
        pass

    def buildKernelsMatrices(self):
        n_latents = len(self._kernels)
        self._Kzz = [[None] for k in range(n_latents)]
        self._Kzz_inv = [[None] for k in range(n_latents)]

        for k in range(n_latents):
            self._Kzz[k] = (self._kernels[k].buildKernelMatrix(X1=self._ind_points_locs[k])+
                            self._reg_param*torch.eye(n=self._ind_points_locs[k].shape[1],
                                                      dtype=self._ind_points_locs[k].dtype,
                                                      device=self._ind_points_locs[k].device))
            self._Kzz_inv[k] = self._invertKzz3D(self._Kzz[k]) # O(n^3)

    def getKzz(self):
        return self._Kzz

    def getRegParam(self):
        return self._reg_param


class IndPointsLocsKMS_Chol(IndPointsLocsKMS):

    def _invertKzz3D(self, Kzz):
        Kzz_inv = svGPFA.utils.miscUtils.chol3D(Kzz)  # O(n^3)
        return Kzz_inv

    def solveForLatent(self, input, latentIndex):
        solve = torch.cholesky_solve(input, self._Kzz_inv[latentIndex])
        return solve

    def solveForLatentAndTrial(self, input, latentIndex, trialIndex):
        solve = torch.cholesky_solve(input, self._Kzz_inv[latentIndex][trialIndex, :, :])
        return solve


class IndPointsLocsKMS_CholWithGettersAndSetters(IndPointsLocsKMS_Chol, KernelMatricesStoreGettersAndSetters):
    def __init__(self):
        pass


class IndPointsLocsKMS_PInv(IndPointsLocsKMS):

    def _invertKzz3D(self, Kzz):
        Kzz_inv = svGPFA.utils.miscUtils.pinv3D(Kzz)  # O(n^3)
        return Kzz_inv

    def solveForLatent(self, input, latentIndex):
        solve = torch.matmul(self._Kzz_inv[latentIndex], input)
        return solve

    def solveForLatentAndTrial(self, input, latentIndex, trialIndex):
        solve = torch.matmul(self._Kzz_inv[latentIndex][trialIndex, :, :],
                             input)
        return solve


class IndPointsLocsKMS_PInvWithGettersAndSetters(IndPointsLocsKMS_PInv,
                                                 KernelMatricesStoreGettersAndSetters):
    def __init__(self):
        pass


class IndPointsLocsAndTimesKMS(KernelsMatricesStore):

    def setTimes(self, times):
        # times \in nTrials x nQuad x 1
        self._t = times

    def getKtz(self):
        return self._Ktz

    def getKtt(self):
        return self._Ktt

    def getKttDiag(self):
        return self._KttDiag


class IndPointsLocsAndAllTimesKMS(IndPointsLocsAndTimesKMS):

    def buildKernelsMatrices(self):
        # self._t \in nTrials x nQuad x 1
        n_latents = len(self._ind_points_locs)
        self._Ktz = [[None] for k in range(n_latents)]
        self._KttDiag = torch.zeros(self._t.shape[0], self._t.shape[1],
                                    n_latents,
                                    dtype=self._t.dtype, device=self._t.device)
        for k in range(n_latents):
            self._Ktz[k] = self._kernels[k].buildKernelMatrix(X1=self._t, X2=self._ind_points_locs[k])
            self._KttDiag[:, :, k] = self._kernels[k].buildKernelMatrixDiag(X=self._t).squeeze()

    def buildKttKernelsMatrices(self):
        # t \in nTrials x nQuad x 1
        n_latents = len(self._ind_points_locs)
        self._Ktt = [[None] for k in range(n_latents)]

        for k in range(n_latents):
            self._Ktt[k] = self._kernels[k].buildKernelMatrix(X1=self._t, X2=self._t)


class IndPointsLocsAndAssocTimesKMS(IndPointsLocsAndTimesKMS):

    def buildKernelsMatrices(self):
        n_latents = len(self._ind_points_locs)
        n_trials = self._ind_points_locs[0].shape[0]
        self._Ktz = [[[None] for tr in range(n_trials)]
                     for k in range(n_latents)]
        self._KttDiag = [[[None] for tr in range(n_trials)] for k in
                         range(n_latents)]

        for k in range(n_latents):
            for tr in range(n_trials):
                self._Ktz[k][tr] = self._kernels[k].buildKernelMatrix(
                    X1=self._t[tr], X2=self._ind_points_locs[k][tr, :, :])
                self._KttDiag[k][tr] = self._kernels[k].buildKernelMatrixDiag(
                    X=self._t[tr])
