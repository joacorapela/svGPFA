
from . import expectedLogLikelihood, klDivergence


class SVLowerBound:

    def eval(vMean, vCov, C, d, Kzz, Kzz_inv, KtzQuad, KtzSpike, KttDiag=1.0):
        eLL_value = expectedLogLikelihood.PointProcessELLExpLink.evalSumAcrossTrialsAndNeurons(
            vMean=vMean, vCov=vCov, C=C, d=d, Kzz=Kzz, Kzz_inv=Kzz_inv,
            KtzQuad=KtzQuad, KtzSpike=KtzSpike, KttDiag=KttDiag)
        kl_sum = klDivergence.KLDivergence.evalSumAcrossLatentsAndTrials(
            vMean=vMean, vCov=vCov, Kzz=Kzz, Kzz_inv=Kzz_inv)
        the_eval = eLL_value-kl_sum
        return the_eval
