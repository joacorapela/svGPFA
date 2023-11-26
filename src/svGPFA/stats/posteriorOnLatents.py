
from abc import ABC, abstractmethod
import torch
import scipy.stats
import svGPFA.stats.kernelsMatricesStore


class PosteriorOnLatents(ABC):

    def __init__(self, variationalDist, indPointsLocsKMS,
                 indPointsLocsAndTimesKMS):
        self._variationalDist = variationalDist
        self._indPointsLocsKMS = indPointsLocsKMS
        self._indPointsLocsAndTimesKMS = indPointsLocsAndTimesKMS

    def computeMeansAndVars(self):
        Kzz = self._indPointsLocsKMS.getKzz()
        Ktz = self._indPointsLocsAndTimesKMS.getKtz()
        KttDiag = self._indPointsLocsAndTimesKMS.getKttDiag()
        answer = self.__computeMeansAndVarsGivenKernelMatrices(Kzz=Kzz,
                                                               Ktz=Ktz,
                                                               KttDiag=KttDiag)
        return answer

    def __computeMeansAndVarsGivenKernelMatrices(self, Kzz, Ktz, KttDiag):
        nTrials = len(KttDiag[0])
        nLatent = len(self._variationalDist.getMean())
        # Ak[k] \in nTrial x nInd[k] x 1
        Ak = [self._indPointsLocsKMS.solveForLatent(
            input=self._variationalDist.getMean()[k], latentIndex=k)
            for k in range(nLatent)]
        qSigma = self._variationalDist.getCov()
        qKMu = [[None] for tr in range(nTrials)]
        qKVar = [[None] for tr in range(nTrials)]
        for trialIndex in range(nTrials):
            nTimesForTrial = KttDiag[0][trialIndex].shape[0]
            # qKMu[trialIndex] \in nTimesForTrial[trialIndex] x nLatent
            qKMu[trialIndex] = torch.empty((nTimesForTrial, nLatent),
                                           dtype=Kzz[0].dtype,
                                           device=Kzz[0].device)
            qKVar[trialIndex] = torch.empty((nTimesForTrial, nLatent),
                                            dtype=Kzz[0].dtype,
                                            device=Kzz[0].device)
            for k in range(nLatent):
                qKMu[trialIndex][:, k] = torch.squeeze(
                    torch.mm(input=Ktz[k][trialIndex],
                             mat2=Ak[k][trialIndex, :, :]))
                # Bfk \in nInd[k] x nTimesForTrial[trialIndex]
                Bfk = self._indPointsLocsKMS.solveForLatentAndTrial(
                    input=Ktz[k][trialIndex].transpose(dim0=0, dim1=1),
                    latentIndex=k, trialIndex=trialIndex)
                # mm1f \in nInd[k] x nTimesForTrial[trialIndex]
                diff = qSigma[k][trialIndex, :, :]-Kzz[k][trialIndex, :, :]
                mm1f = torch.matmul(diff, Bfk)
                # qKVar[trialIndex] \in nTimesForTrial[trialIndex] x nLatent
                qKVar[trialIndex][:, k] = \
                    torch.squeeze(KttDiag[k][trialIndex]) + \
                    torch.sum(a=Bfk*mm1f, axis=0)

        return qKMu, qKVar

    def setTimes(self, times):
        self._indPointsLocsAndTimesKMS.setTimes(times=times)

    def getVariationalDistParams(self):
        return self._variationalDist.getParams()

    def getKernels(self):
        return self._indPointsLocsKMS.getKernels()

    def getKernelsParams(self):
        return self._indPointsLocsKMS.getKernelsParams()

    def getIndPointsLocs(self):
        return self._indPointsLocsKMS.getIndPointsLocs()

    @abstractmethod
    def buildKernelsMatrices(self):
        pass

    @abstractmethod
    def setKernels(self, kernels):
        pass

    @abstractmethod
    def setInitialParams(self, initial_params):
        pass

    @abstractmethod
    def setIndPointsLocs(self, indPointsLocs):
        pass


class PosteriorOnLatentsQuadTimes(PosteriorOnLatents):

    def predict(self, times):
        # times \in n_trials x n_times x 1
        # Note: forcing all trials to have the same number of time samples is a
        # limitation of the current version of the code

        kernels = self._indPointsLocsKMS.getKernels()

        indPointsLocsAndAllTimesKMS = \
            svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
        indPointsLocsAndAllTimesKMS.setKernels(kernels=kernels)
        indPointsLocsAndAllTimesKMS.setTimes(times=times)
        indPointsLocsAndAllTimesKMS.setIndPointsLocs(
            ind_points_locs=self.getIndPointsLocs())
        indPointsLocsAndAllTimesKMS.buildKernelsMatrices()

        svPosteriorOnLatents = PosteriorOnLatentsAllTimes(
            variationalDist=self._variationalDist,
            indPointsLocsKMS=self._indPointsLocsKMS,
            indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
        qKMu, qKVar = svPosteriorOnLatents.computeMeansAndVars()
        return qKMu, qKVar

    def computeMeansAndVarsAtTimes(self, times):
        Kzz = self._indPointsLocsKMS.getKzz()

        indPointsLocsAndAllTimesKMS = \
            svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
        indPointsLocsAndAllTimesKMS.setKernels(
            kernels=self._indPointsLocsKMS.getKernels())
        indPointsLocsAndAllTimesKMS.setTimes(times=times)
        indPointsLocsAndAllTimesKMS.setIndPointsLocs(
            indPointsLocs=self.getIndPointsLocs())
        indPointsLocsAndAllTimesKMS.buildKernelsMatrices()

        Ktz = indPointsLocsAndAllTimesKMS.getKtz()
        KttDiag = indPointsLocsAndAllTimesKMS.getKttDiag()
        answer = self.__computeMeansAndVarsGivenKernelMatrices(Kzz=Kzz,
                                                               Ktz=Ktz,
                                                               KttDiag=KttDiag)

        return answer

    def sample(self, times, nudget=1e-3):
        Kzz = self._indPointsLocsKMS.getKzz()
        KzzInv = self._indPointsLocsKMS.getKzzInv()

        indPointsLocsAndAllTimesKMS = \
            svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
        indPointsLocsAndAllTimesKMS.setKernels(
            kernels=self._indPointsLocsKMS.getKernels())
        indPointsLocsAndAllTimesKMS.setIndPointsLocs(
            indPointsLocs=self._indPointsLocsKMS.getIndPointsLocs())
        indPointsLocsAndAllTimesKMS.setTimes(times=times)
        indPointsLocsAndAllTimesKMS.buildKernelsMatrices()
        indPointsLocsAndAllTimesKMS.buildKttKernelsMatrices()
        Ktz = indPointsLocsAndAllTimesKMS.getKtz()
        Ktt = indPointsLocsAndAllTimesKMS.getKtt()

        qMu = self._variationalDist.getMean()
        qSigma = self._variationalDist.getCov()

        nLatents = len(Kzz)
        nTrials = Kzz[0].shape[0]
        samples = [[] for r in range(nTrials)]
        means = [[] for r in range(nTrials)]
        variances = [[] for r in range(nTrials)]
        for r in range(nTrials):
            samples[r] = torch.empty((nLatents, Ktt[0].shape[1]),
                                     dtype=Kzz[0].dtype)
            means[r] = torch.empty((nLatents, Ktt[0].shape[1]),
                                   dtype=Kzz[0].dtype)
            variances[r] = torch.empty((nLatents, Ktt[0].shape[1]),
                                       dtype=Kzz[0].dtype)
            for k in range(nLatents):
                print("Processing trial {:d} and latent {:d}".format(r, k))
                Kzzrk = Kzz[k][r, :, :]
                KzzInvrk = KzzInv[k][r,:,:]
                Ktzrk = Ktz[k][r, :, :]
                Kttrk = Ktt[k][r, :, :]
                qMurk = qMu[k][r, :, :]
                qSigmark = qSigma[k][r, :, :]

                # begin compute mean #
                # b = torch.cholesky_solve(qMurk, KzzInvrk)
                b = self._indPointsLocsKMS.solveForLatentAndTrial(
                    input=qMurk, latentIndex=k, trialIndex=r)
                meanrk = torch.squeeze(Ktzrk.matmul(b)).detach().numpy()
                # end compute mean #

                # being compute covar #
                # B = torch.cholesky_solve(torch.t(Ktzrk), KzzInvrk)
                B = self._indPointsLocsKMS.solveForLatentAndTrial(
                    input=torch.t(Ktzrk), latentIndex=k, trialIndex=r)
                covarrk = Kttrk+torch.t(B).matmul(qSigmark-Kzzrk).matmul(B)
                # end compute covar #

                covarrk += torch.eye(covarrk.shape[0])*nudget
                covarrk = covarrk.detach().numpy()
                mn = scipy.stats.multivariate_normal(mean=meanrk, cov=covarrk)
                samples[r][k, :] = torch.from_numpy(mn.rvs(size=1))
                means[r][k, :] = torch.from_numpy(meanrk)
                variances[r][k, :] = torch.diag(torch.from_numpy(covarrk))
        return samples, means, variances

    def buildVariationalCov(self):
        self._variationalDist.buildCov()

    def setPriorCovRegParam(self, priorCovRegParam):
        self._indPointsLocsKMS.setRegParam(reg_param=priorCovRegParam)

    def buildKernelsMatrices(self):
        self._indPointsLocsKMS.buildKernelsMatrices()
        self._indPointsLocsAndTimesKMS.buildKernelsMatrices()

    def setKernels(self, kernels):
        self._indPointsLocsKMS.setKernels(kernels=kernels)
        self._indPointsLocsAndTimesKMS.setKernels(kernels=kernels)

    def setInitialParams(self, initial_params):
        self._variationalDist.setInitialParams(
            initial_params=initial_params["posterior_on_ind_points"])
        self._indPointsLocsKMS.setInitialParams(
            initial_params=initial_params["kernels_matrices_store"])
        self._indPointsLocsAndTimesKMS.setInitialParams(
            initial_params=initial_params["kernels_matrices_store"])

    def setIndPointsLocs(self, indPointsLocs):
        self._indPointsLocsKMS.setIndPointsLocs(ind_pointsLocs=indPointsLocs)
        self._indPointsLocsAndTimesKMS.setIndPointsLocs(
            ind_points_locs=indPointsLocs)


class PosteriorOnLatentsSpikesTimes(PosteriorOnLatents):

    def buildKernelsMatrices(self):
        # not asking _indPointsLocsKMS to buildMatrices because
        # SVPosteriorOnLatentsQuadTimes already did so
        self._indPointsLocsAndTimesKMS.buildKernelsMatrices()

    def setKernels(self, kernels):
        # not asking _indPointsLocssKMS to setKernels because
        # SVPosteriorOnLatentsQuadTimes already did so
        self._indPointsLocsAndTimesKMS.setKernels(kernels=kernels)

    def setInitialParams(self, initial_params):
        # not asking _indPointsLocsKMS to setInitialParams becasue
        # SVPosteriorOnLatentsAlTimes already did so
        self._indPointsLocsAndTimesKMS.setInitialParams(
            initial_params=initial_params["kernels_matrices_store"])

    def setIndPointsLocs(self, indPointsLocs):
        # not asking _indPointsLocsKMS to setIndPointsLocs becasue
        # SVPosteriorOnLatentsAlTimes already did so
        self._indPointsLocsAndTimesKMS.setIndPointsLocs(
            ind_points_locs=indPointsLocs)
