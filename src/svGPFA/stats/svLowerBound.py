
from . import expectedLogLikelihood, klDivergence


class SVLowerBound:

    def eval(vMean, vCov, C, d, Kzz, Kzz_cho, KtzQuad, KtzSpikes, KttDiag=1.0):
        eLL_value = expectedLogLikelihood.PointProcessELLExpLink.evalSumAcrossTrialsAndNeurons(
            vMean=vMean, vCov=vCov, C=C, d=d, Kzz=Kzz, Kzz_cho=Kzz_cho,
            KtzQuad=KtzQuad, KtzSpikes=KtzSpikes, KttDiag=KttDiag)
        kl_sum = klDivergence.KLDivergence.evalSumAcrossLatentsAndTrials(
            vMean=vMean, vCov=vCov, Kzz=Kzz, Kzz_cho=Kzz_cho)
        the_eval = eLL_value-kl_sum
        print(f"eLL_value={eLL_value}")
        print(f"kl_sum={kl_sum}")
        return the_eval
