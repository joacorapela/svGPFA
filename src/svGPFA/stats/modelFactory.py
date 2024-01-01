
import svGPFA.stats.kernelsMatricesStore
import svGPFA.stats.variationalDist
import svGPFA.stats.posteriorOnLatents
import svGPFA.stats.preIntensity
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
LinearPreIntensity = 100

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


class ModelFactory:

    @staticmethod
    def buildModelJAX(kernels, legQuadWeights,
                      neuronForSpikeIndex,
                      conditionalDist=PointProcess,
                      linkFunction=ExponentialLink,
                      preIntensityType=LinearPreIntensity,
                      indPointsCovRep=indPointsCovChol):
        """Creates an svGPFA model. :meth:`svGPFA.stats.svLowerBound.SVLowerBound.setInitialDataAndParams` should be invoked before using the created model as argument to :meth:`svGPFA.stats.svEM.SVEM.maximize`.

        :param kernels: list of kernels (:mod:`svGPFA.stats.kernel`) to be used in the model
        :type kernels: list of instances from subclass of (:class:`svGPFA.stats.kernel.Kernel`)
        :param conditionalDist: likelihood distribution (e.g., svGPFA.stats.ModelFactory.PointProcess or svGPFA.stats.ModelFactory.Gaussian)
        :type conditionalDist: int
        :param preIntensityType: type of preIntensity (e.g., svGPFA.stats.ModelFactory.LinearPreIntensity)
        :type preIntensityType: int

        :return: an unitialized model. Parameters and data need to be set (by calling :meth:`svGPFA.stats.svLowerBound.SVLowerBound.setParamsAndData`) before invoking :meth:`svGPFA.stats.svEM.SVEM.maximize`.
        :rtype: an instance of :class:`svGPFA.stats.svLowerBound.SVLowerBound`.

        """

        if conditionalDist == PointProcess:
            if preIntensityType == LinearPreIntensity:
                if linkFunction == ExponentialLink:
                    if indPointsCovRep == indPointsCovChol:
                        pass
#                         qU = svGPFA.stats.variationalDist.\
#                                 VariationalDistChol()
                    elif indPointsCovRep == indPointsCovRank1PlusDiag:
                        raise NotImplmentedError()
                    else:
                        raise ValueError("Invalid indPointsCovRep")
                    qK = svGPFA.stats.posteriorOnLatents.PosteriorOnLatents()
                    qHQuadTimes = svGPFA.stats.preIntensity.\
                        LinearPreIntensityQuadTimes(posteriorOnLatents=qK)
                    qHSpikesTimes = svGPFA.stats.preIntensity.\
                        LinearPreIntensitySpikesTimes(
                            posteriorOnLatents=qK,
                            neuronForSpikeIndex=neuronForSpikeIndex)
                    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(
                        preIntensityQuadTimes=qHQuadTimes,
                        preIntensitySpikesTimes=qHSpikesTimes,
                        legQuadWeights=legQuadWeights,
                    )
                    klDiv = svGPFA.stats.klDivergence.KLDivergence()
                    svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL,
                                                                  klDiv=klDiv)
                else:
                    raise ValueError("Invalid linkFunction={:s}".
                                     format(repr(linkFunction)))
            else:
                raise ValueError("Invalid preIntensityType={:s}".
                                 format(repr(preIntensityType)))
        else:
            raise ValueError("Invalid conditionalDist=%s"%
                             repr(conditionalDist))

        return svlb

    @staticmethod
    def buildModelPyTorch(kernels, conditionalDist=PointProcess,
                          linkFunction=ExponentialLink,
                          preIntensityType=LinearPreIntensity,
                          kernelMatrixInvMethod=kernelMatrixInvChol,
                          indPointsCovRep=indPointsCovChol):
        """Creates an svGPFA model. :meth:`svGPFA.stats.svLowerBound.SVLowerBound.setInitialDataAndParams` should be invoked before using the created model as argument to :meth:`svGPFA.stats.svEM.SVEM.maximize`.

        :param kernels: list of kernels (:mod:`svGPFA.stats.kernel`) to be used in the model
        :type kernels: list of instances from subclass of (:class:`svGPFA.stats.kernel.Kernel`)
        :param conditionalDist: likelihood distribution (e.g., svGPFA.stats.ModelFactory.PointProcess or svGPFA.stats.ModelFactory.Gaussian)
        :type conditionalDist: int
        :param preIntensityType: type of preIntensity (e.g., svGPFA.stats.ModelFactory.LinearPreIntensity)
        :type preIntensityType: int

        :return: an unitialized model. Parameters and data need to be set (by calling :meth:`svGPFA.stats.svLowerBound.SVLowerBound.setParamsAndData`) before invoking :meth:`svGPFA.stats.svEM.SVEM.maximize`.
        :rtype: an instance of :class:`svGPFA.stats.svLowerBound.SVLowerBound`.

        """

        if conditionalDist == PointProcess:
            if preIntensityType == LinearPreIntensity:
                if linkFunction == ExponentialLink:
                    if indPointsCovRep == indPointsCovChol:
                        qU = svGPFA.stats.variationalDist.\
                                VariationalDistChol()
                    elif indPointsCovRep == indPointsCovRank1PlusDiag:
                        qU = svGPFA.stats.variationalDist.\
                                VariationalDistRank1PlusDiag()
                    else:
                        raise ValueError("Invalid indPointsCovRep")
                    if kernelMatrixInvMethod == kernelMatrixInvChol:
                        indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.\
                                IndPointsLocsKMS_Chol()
                    elif kernelMatrixInvMethod == kernelMatrixInvPInv:
                        indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.\
                                IndPointsLocsKMS_PInv()
                    else:
                        raise ValueError("Invalid kernelMatrixInvMethod")
                    indPointsLocsAndQuadTimesKMS = svGPFA.stats.\
                        kernelsMatricesStore.IndPointsLocsAndTimesKMS()
                    indPointsLocsAndSpikesTimesKMS = svGPFA.stats.\
                        kernelsMatricesStore.IndPointsLocsAndTimesKMS()
                    qKQuadTimes = svGPFA.stats.posteriorOnLatents.\
                        PosteriorOnLatentsQuadTimes(
                            variationalDist=qU,
                            indPointsLocsKMS=indPointsLocsKMS,
                            indPointsLocsAndTimesKMS=indPointsLocsAndQuadTimesKMS)
                    qKSpikesTimes = svGPFA.stats.posteriorOnLatents.\
                        PosteriorOnLatentsSpikesTimes(
                            variationalDist=qU,
                            indPointsLocsKMS=indPointsLocsKMS,
                            indPointsLocsAndTimesKMS=indPointsLocsAndSpikesTimesKMS)
                    qHQuadTimes = svGPFA.stats.preIntensity.\
                        LinearPreIntensityQuadTimes(
                            posteriorOnLatents=qKQuadTimes)
                    qHSpikesTimes = svGPFA.stats.preIntensity.\
                        LinearPreIntensitySpikesTimes(
                            posteriorOnLatents=qKSpikesTimes)
                    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(
                        preIntensityQuadTimes=qHQuadTimes,
                        preIntensitySpikesTimes=qHSpikesTimes)
                    klDiv = svGPFA.stats.klDivergence.KLDivergence(
                        indPointsLocsKMS=indPointsLocsKMS,
                        variationalDist=qU)
                    svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL,
                                                                  klDiv=klDiv)
                    svlb.setKernels(kernels=kernels)
                else:
                    raise ValueError("Invalid linkFunction={:s}".
                                     format(repr(linkFunction)))
            else:
                raise ValueError("Invalid preIntensityType={:s}".
                                 format(repr(preIntensityType)))
        else:
            raise ValueError("Invalid conditionalDist=%s"%
                             repr(conditionalDist))

        return svlb

    @staticmethod
    def buildModelSciPy(conditionalDist, linkFunction, preIntensityType, kernels,
                        kernelMatrixInvMethod, indPointsCovRep):

        if conditionalDist == PointProcess:
            if preIntensityType == LinearPreIntensity:
                if linkFunction == ExponentialLink:
                    if indPointsCovRep == indPointsCovChol:
                        qU = svGPFA.stats.variationalDist.\
                            VariationalDistCholWithGettersAndSetters()
                    elif indPointsCovRep == indPointsCovRank1PlusDiag:
                        qU = svGPFA.stats.variationalDist.\
                            VariationalDistRank1PlusDiagWithGettersAndSetters()
                    else:
                        raise ValueError("Invalid indPointsCovRep")
                    if kernelMatrixInvMethod == kernelMatrixInvChol:
                        indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.\
                            IndPointsLocsKMS_CholWithGettersAndSetters()
                    elif kernelMatrixInvMethod == kernelMatrixInvPInv:
                        indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.\
                            IndPointsLocsKMS_PInvWithGettersAndSetters()
                    else:
                        raise ValueError("Invalid kernelMatrixInvMethod")
                    indPointsLocsAndQuadTimesKMS = svGPFA.stats.\
                        kernelsMatricesStore.IndPointsLocsAndTimesKMS()
                    indPointsLocsAndSpikesTimesKMS = svGPFA.stats.\
                        kernelsMatricesStore.IndPointsLocsAndTimesKMS()
                    qKQuadTimes = svGPFA.stats.posteriorOnLatents.\
                        PosteriorOnLatentsQuadTimes(
                            variationalDist=qU,
                            indPointsLocsKMS=indPointsLocsKMS,
                            indPointsLocsAndTimesKMS=indPointsLocsAndQuadTimesKMS)
                    qKSpikesTimes = svGPFA.stats.posteriorOnLatents.\
                            PosteriorOnLatentsSpikesTimes(
                                variationalDist=qU,
                                indPointsLocsKMS=indPointsLocsKMS,
                                indPointsLocsAndTimesKMS=indPointsLocsAndSpikesTimesKMS)
                    qHQuadTimes = svGPFA.stats.preIntensity.\
                            LinearPreIntensityQuadTimesWithParamsGettersAndSetters(
                                posteriorOnLatents=qKQuadTimes)
                    qHSpikesTimes = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(
                        posteriorOnLatents=qKSpikesTimes)
                    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(
                        preIntensityQuadTimes=qHQuadTimes,
                        preIntensitySpikesTimes=qHSpikesTimes)
                    klDiv = svGPFA.stats.klDivergence.KLDivergence(
                        indPointsLocsKMS=indPointsLocsKMS,
                        variationalDist=qU)
                    svlb = svGPFA.stats.svLowerBound.\
                        SVLowerBoundWithParamsGettersAndSetters(eLL=eLL,
                                                                klDiv=klDiv)
                    svlb.setKernels(kernels=kernels)
                else:
                    raise ValueError("Invalid linkFunction={:s}".
                                     format(repr(linkFunction)))
            else:
                raise ValueError("Invalid preIntensityType={:s}".
                                 format(repr(preIntensityType)))
        else:
            raise ValueError("Invalid conditionalDist={:s}".
                             format(repr(conditionalDist)))

        return svlb
