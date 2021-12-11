
import stats.svGPFA.kernelsMatricesStore
import stats.svGPFA.svPosteriorOnIndPoints
import stats.svGPFA.svPosteriorOnLatents
import stats.svGPFA.svEmbedding
import stats.svGPFA.expectedLogLikelihood
import stats.svGPFA.klDivergence
import stats.svGPFA.svLowerBound

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
    def buildModelPytorch(conditionalDist, linkFunction, embeddingType, kernels,
                   kernelMatrixInvMethod, indPointsCovRep):

        if conditionalDist==PointProcess:
            if embeddingType==LinearEmbedding:
                if linkFunction==ExponentialLink:
                    if indPointsCovRep==indPointsCovChol:
                        qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
                    elif indPointsCovRep==indPointsCovRank1PlusDiag:
                        qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPointsRank1PlusDiag()
                    else:
                        raise ValueError("Invalid indPointsCovRep")
                    if kernelMatrixInvMethod==kernelMatrixInvChol:
                        indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS_Chol()
                    elif kernelMatrixInvMethod==kernelMatrixInvPInv:
                        indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS_PInv()
                    else:
                        raise ValueError("Invalid kernelMatrixInvMethod")
                    indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
                    indPointsLocsAndAssocTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAssocTimesKMS()
                    qKAllTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
                    qKAssocTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
                    qHAllTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAllTimes(
                        svPosteriorOnLatents=qKAllTimes)
                    qHAssocTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAssocTimes(
                        svPosteriorOnLatents=qKAssocTimes)
                    eLL = stats.svGPFA.expectedLogLikelihood.PointProcessELLExpLink(
                        svEmbeddingAllTimes=qHAllTimes,
                        svEmbeddingAssocTimes=qHAssocTimes)
                    klDiv = stats.svGPFA.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                                         svPosteriorOnIndPoints=qU)
                    svlb = stats.svGPFA.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
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
                    qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
                    if kernelMatrixInvMethod==Cholesky:
                        indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS_Chol()
                    elif kernelMatrixInvMethod==PInv:
                        indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS_Pinv()
                    else:
                        raise ValueError("Invalid kernelMatrixInvMethod")
                    indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
                    qKAllTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
                    qHAllTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAllTimes(
                        svPosteriorOnLatents=qKAllTimes)
                    eLL = stats.svGPFA.expectedLogLikelihood.PoissonELLExpLink(
                        svEmbeddingAllTimes=qHAllTimes)
                    klDiv = stats.svGPFA.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                                         svPosteriorOnIndPoints=qU)
                    svlb = stats.svGPFA.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
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
                        qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPointsCholWithGettersAndSetters()
                    elif indPointsCovRep==indPointsCovRank1PlusDiag:
                        qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPointsRank1PlusDiagWithGettersAndSetters()
                    else:
                        raise ValueError("Invalid indPointsCovRep")
                    if kernelMatrixInvMethod==kernelMatrixInvChol:
                        indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS_CholWithGettersAndSetters()
                    elif kernelMatrixInvMethod==kernelMatrixInvPInv:
                        indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS_PInvWithGettersAndSetters()
                    else:
                        raise ValueError("Invalid kernelMatrixInvMethod")
                    indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
                    indPointsLocsAndAssocTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAssocTimesKMS()
                    qKAllTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
                    qKAssocTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
                    qHAllTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAllTimesWithParamsGettersAndSetters(
                        svPosteriorOnLatents=qKAllTimes)
                    qHAssocTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAssocTimes(
                        svPosteriorOnLatents=qKAssocTimes)
                    eLL = stats.svGPFA.expectedLogLikelihood.PointProcessELLExpLink(
                        svEmbeddingAllTimes=qHAllTimes,
                        svEmbeddingAssocTimes=qHAssocTimes)
                    klDiv = stats.svGPFA.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                                         svPosteriorOnIndPoints=qU)
                    svlb = stats.svGPFA.svLowerBound.SVLowerBoundWithParamsGettersAndSetters(eLL=eLL, klDiv=klDiv)
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
                    qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
                    if kernelMatrixInvMethod==Cholesky:
                        indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS_Chol()
                    elif kernelMatrixInvMethod==PInv:
                        indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS_Pinv()
                    else:
                        raise ValueError("Invalid kernelMatrixInvMethod")
                    indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
                    qKAllTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
                    qHAllTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAllTimes(
                        svPosteriorOnLatents=qKAllTimes)
                    eLL = stats.svGPFA.expectedLogLikelihood.PoissonELLExpLink(
                        svEmbeddingAllTimes=qHAllTimes)
                    klDiv = stats.svGPFA.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                                         svPosteriorOnIndPoints=qU)
                    svlb = stats.svGPFA.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
                    svlb.setKernels(kernels=kernels)
        else:
            raise ValueError("Invalid conditionalDist=%s"%
                             repr(conditionalDist))

        return svlb

