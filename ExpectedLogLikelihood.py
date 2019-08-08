
from abc import ABCMeta

class ExpectedLogLikelihood(ABCMeta):
    '''

    Abstract base class for expected log-likelihood subclasses 
    (e.g., PointProcessExpectedLogLikelihood).


    '''

    def __init__(data, quadPoints, quadWeights, spikeTimes):
        '''

        Parameters
        ----------
        data : array
               observations

        quadPoints : array
                     points x_i's used to compute the expectation integral
                     by Gauss-Hermite quadrature.

        quadWeights : array
                      weights w_i's used to compute the expectation
                      integral by Gauss-Hermite quadrature.

        spikeTimes : array
                     array of spike times

        '''
        pass

    def evalWithGradientOnQ(qMean, qVar):
        '''Evaluates the expected log likelihood of a svGPFA

        Parameters
        ----------
        qMean : array
                mean of q(h_n), as computed by
                SparseVariationalLowerBound::computeQ()

        qVar : array
               variance of q(h_n), as computed by
               SparseVariationalLowerBound::computeQ()

        Returns
        -------
        value : double
                value of expectation

        gradient: array
                  gradient of expectation wrt mean and covariance of q

        '''
        pass

