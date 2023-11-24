
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
        # input \in (nTrials, nIndPoints[k], 1)
        # self._Kzz_inv \ in (nTrials, nIndPoints[k], nIndPoints[k])
        solve = torch.cholesky_solve(input, self._Kzz_inv[latentIndex])
        return solve

    def solveForLatentAndTrial(self, input, latentIndex, trialIndex):
        solve = torch.cholesky_solve(input, self._Kzz_inv[latentIndex][trialIndex, :, :])
        return solve


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


class IndPointsLocsAndTimesKMS(KernelsMatricesStore):

    def setTimes(self, times):
        # times[r] \in nTimes x 1
        self._t = times

    def getKtz(self):
        return self._Ktz

    def getKtt(self):
        return self._Ktt

    def getKttDiag(self):
        return self._KttDiag

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
