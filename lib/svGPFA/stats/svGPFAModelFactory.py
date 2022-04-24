
import svGPFA.stats.kernelsMatricesStore
import svGPFA.stats.svPosteriorOnIndPoints
import svGPFA.stats.svPosteriorOnLatents
import svGPFA.stats.svEmbedding
import svGPFA.stats.expectedLogLikelihood
import svGPFA.stats.klDivergence
import svGPFA.stats.svLowerBound

#:
PointProcess = 0
#:
Poisson = 1
#:
Gaussian = 2


#:
LinearEmbedding = 100

#:
ExponentialLink = 1000
#:
NonExponentialLink = 1001

#:
kernelMatrixInvChol = 10000
#:
kernelMatrixInvPInv = 10001

#:
indPointsCovRank1PlusDiag = 100000
#:
indPointsCovChol = 100001

class SVGPFAModelFactory:

    @staticmethod
    def buildModelPyTorch(conditionalDist, linkFunction, embeddingType, kernels,
                   kernelMatrixInvMethod, indPointsCovRep):

        if conditionalDist==PointProcess:
            if embeddingType==LinearEmbedding:
                if linkFunction==ExponentialLink:
                    if indPointsCovRep==indPointsCovChol:
                        qU = svGPFA.stats.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
                    elif indPointsCovRep==indPointsCovRank1PlusDiag:
                        qU = svGPFA.stats.svPosteriorOnIndPoints.SVPosteriorOnIndPointsRank1PlusDiag()
                    else:
                        raise ValueError("Invalid indPointsCovRep")
                    if kernelMatrixInvMethod==kernelMatrixInvChol:
                        indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
                    elif kernelMatrixInvMethod==kernelMatrixInvPInv:
                        indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_PInv()
                    else:
                        raise ValueError("Invalid kernelMatrixInvMethod")
                    indPointsLocsAndAllTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
                    indPointsLocsAndAssocTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAssocTimesKMS()
                    qKAllTimes = svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
                    qKAssocTimes = svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
                    qHAllTimes = svGPFA.stats.svEmbedding.LinearSVEmbeddingAllTimes(
                        svPosteriorOnLatents=qKAllTimes)
                    qHAssocTimes = svGPFA.stats.svEmbedding.LinearSVEmbeddingAssocTimes(
                        svPosteriorOnLatents=qKAssocTimes)
                    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(
                        svEmbeddingAllTimes=qHAllTimes,
                        svEmbeddingAssocTimes=qHAssocTimes)
                    klDiv = svGPFA.stats.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                                         svPosteriorOnIndPoints=qU)
                    svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
                    svlb.setKernels(kernels=kernels)
                else:
                    raise ValueError("Invalid linkFunction=%s"%
                                     repr(linkFunction))
            else:
                raise ValueError("Invalid embeddingType=%s"%
                                 repr(embeddingType))
        elif conditionalDist==Poisson:
            if embeddingType==LinearEmbedding:
                if linkFunction==ExponentialLink:
                    qU = svGPFA.stats.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
                    if kernelMatrixInvMethod==Cholesky:
                        indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
                    elif kernelMatrixInvMethod==PInv:
                        indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Pinv()
                    else:
                        raise ValueError("Invalid kernelMatrixInvMethod")
                    indPointsLocsAndAllTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
                    qKAllTimes = svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
                    qHAllTimes = svGPFA.stats.svEmbedding.LinearSVEmbeddingAllTimes(
                        svPosteriorOnLatents=qKAllTimes)
                    eLL = svGPFA.stats.expectedLogLikelihood.PoissonELLExpLink(
                        svEmbeddingAllTimes=qHAllTimes)
                    klDiv = svGPFA.stats.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                                         svPosteriorOnIndPoints=qU)
                    svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
                    svlb.setKernels(kernels=kernels)
        else:
            raise ValueError("Invalid conditionalDist=%s"%
                             repr(conditionalDist))

        return svlb

    @staticmethod
    def buildModelSciPy(conditionalDist, linkFunction, embeddingType, kernels,
                        kernelMatrixInvMethod, indPointsCovRep):

        if conditionalDist==PointProcess:
            if embeddingType==LinearEmbedding:
                if linkFunction==ExponentialLink:
                    if indPointsCovRep==indPointsCovChol:
                        qU = svGPFA.stats.svPosteriorOnIndPoints.SVPosteriorOnIndPointsCholWithGettersAndSetters()
                    elif indPointsCovRep==indPointsCovRank1PlusDiag:
                        qU = svGPFA.stats.svPosteriorOnIndPoints.SVPosteriorOnIndPointsRank1PlusDiagWithGettersAndSetters()
                    else:
                        raise ValueError("Invalid indPointsCovRep")
                    if kernelMatrixInvMethod==kernelMatrixInvChol:
                        indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_CholWithGettersAndSetters()
                    elif kernelMatrixInvMethod==kernelMatrixInvPInv:
                        indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_PInvWithGettersAndSetters()
                    else:
                        raise ValueError("Invalid kernelMatrixInvMethod")
                    indPointsLocsAndAllTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
                    indPointsLocsAndAssocTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAssocTimesKMS()
                    qKAllTimes = svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
                    qKAssocTimes = svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
                    qHAllTimes = svGPFA.stats.svEmbedding.LinearSVEmbeddingAllTimesWithParamsGettersAndSetters(
                        svPosteriorOnLatents=qKAllTimes)
                    qHAssocTimes = svGPFA.stats.svEmbedding.LinearSVEmbeddingAssocTimes(
                        svPosteriorOnLatents=qKAssocTimes)
                    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(
                        svEmbeddingAllTimes=qHAllTimes,
                        svEmbeddingAssocTimes=qHAssocTimes)
                    klDiv = svGPFA.stats.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                                         svPosteriorOnIndPoints=qU)
                    svlb = svGPFA.stats.svLowerBound.SVLowerBoundWithParamsGettersAndSetters(eLL=eLL, klDiv=klDiv)
                    svlb.setKernels(kernels=kernels)
                else:
                    raise ValueError("Invalid linkFunction=%s"%
                                     repr(linkFunction))
            else:
                raise ValueError("Invalid embeddingType=%s"%
                                 repr(embeddingType))
        elif conditionalDist==Poisson:
            if embeddingType==LinearEmbedding:
                if linkFunction==ExponentialLink:
                    qU = svGPFA.stats.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
                    if kernelMatrixInvMethod==Cholesky:
                        indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
                    elif kernelMatrixInvMethod==PInv:
                        indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Pinv()
                    else:
                        raise ValueError("Invalid kernelMatrixInvMethod")
                    indPointsLocsAndAllTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
                    qKAllTimes = svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
                    qHAllTimes = svGPFA.stats.svEmbedding.LinearSVEmbeddingAllTimes(
                        svPosteriorOnLatents=qKAllTimes)
                    eLL = svGPFA.stats.expectedLogLikelihood.PoissonELLExpLink(
                        svEmbeddingAllTimes=qHAllTimes)
                    klDiv = svGPFA.stats.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                                         svPosteriorOnIndPoints=qU)
                    svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
                    svlb.setKernels(kernels=kernels)
        else:
            raise ValueError("Invalid conditionalDist=%s"%
                             repr(conditionalDist))

        return svlb

