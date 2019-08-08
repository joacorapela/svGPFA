
import ExpectedLogLikelihood

class PointProcessExpectedLogLikelihood(ExpectedLogLikelihood):

    def __init__(legQuadPoints, legQuadWeights, hermQuadPoints, hermQuadWeights,
                                linkFunction):
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
        self.__legQuadPoints = legQuadPoints
        self.__legQuadWeights = legQuadWeights
        self.__hermQuadPoints = hermQuadPoints
        self.__hermQuadWeights = hermQuadWeights
        self.__linkFunction = linkFunction

    def evalWithGradientOnQ(qHMeanQuad, qHVarQuad, qHMeanSpikes, qHVarSpikes,
                                        trials):
        ''' Evaluates the expected log likelihood of a svGPFA

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

        if self.__linkFunction == self.exp:
            intval = np.exp(mu_h_Quad + 0.5*var_h_Quad)
        else:
            # intval = permute(mtimesx(m.wwHerm',permute(m.link(mu_h_Quad + sqrt(2*var_h_Quad).*permute(m.xxHerm,[2 3 4 1])),[4 1 2 3])),[2 3 4 1]);

            aux1 = np.reshape(self.__hermQuadPoints, 
                               [1,1,1,len(self.__hermQuadPoints)])
            aux2 = np.sqrt(2*qhVarQuad)
            aux3 = np.multiply(aux2, aux1)
            aux4 = np.add(np.reshape(qHMeanQuad, (1,)+aux3.shape)
            aux5 = self.__linkFunction(x=aux4)
            intval = np.tensordot(a=aux5, b=self.__hermQuadWeights, 
                                          axes=([4], [0]))
        return ell

