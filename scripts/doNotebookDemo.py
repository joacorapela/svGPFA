import sys
import pdb
import torch
import pickle

sys.path.append("../src")
import stats.kernels
import stats.svGPFA.svGPFAModelFactory
import stats.svGPFA.svEM
import utils.svGPFA.miscUtils
import utils.svGPFA.initUtils


def main(argv):
    nLatents = 2                                      # number of latents
    nNeurons = 100                                    # number of neurons
    nTrials = 15                                      # number of trials
    nQuad = 200                                       # number of quadrature points
    nIndPoints = 9                                    # number of inducing points
    indPointsLocsKMSRegEpsilon = 1e-5                 # prior covariance nudget parameter
    trial_start_time = 0.0                            # trial start time
    trial_end_time = 1.0                              # trial end time
    lengthscale0 = 1.0                                # initial value of the lengthscale parameter
    simResFilename = "results/32451751_simRes.pickle" # simulation results filename

    # load spikes
    with open(simResFilename, "rb") as f:
        simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]

    # embedding parameters initial values: uniform[0,1]
    C0 = torch.rand(nNeurons, nLatents, dtype=torch.double).contiguous()
    d0 = torch.rand(nNeurons, 1, dtype=torch.double).contiguous()

    # kernels of latents: all ExponentialQuadratic Kernels
    kernels = [[] for r in range(nLatents)]
    for k in range(nLatents):
        kernels[k] = stats.kernels.ExponentialQuadraticKernel()

    # kernels parameters initial values: all kernels have the same initial
    # lengthscale0
    kernelsScaledParams0 = [torch.tensor([lengthscale0], dtype=torch.double)
                            for r in range(nLatents)]

    # inducing points locations initial values: equally spaced nIndPoints
    # between trial_start_time and trial_end_time
    Z0 = [[] for k in range(nLatents)]
    for k in range(nLatents):
        Z0[k] = torch.empty((nTrials, nIndPoints, 1), dtype=torch.double)
        for r in range(nTrials):
            Z0[k][r, :, 0] = torch.linspace(trial_start_time, trial_end_time,
                                            nIndPoints, dtype=torch.double)

    # variational mean initial value: Uniform[0, 1]
    qMu0 = [[] for r in range(nLatents)]
    for k in range(nLatents):
        qMu0[k] = torch.empty((nTrials, nIndPoints, 1), dtype=torch.double)
        for r in range(nTrials):
            qMu0[k][r, :, 0] = torch.rand(nIndPoints)

    # variational covariance initial value: Identity*1e-2
    diag_value = 1e-2
    qSigma0 = [[] for r in range(nLatents)]
    for k in range(nLatents):
        qSigma0[k] = torch.empty((nTrials, nIndPoints, nIndPoints),
                                 dtype=torch.double)
        for r in range(nTrials):
            qSigma0[k][r, :, :] = torch.eye(nIndPoints)*diag_value

    # we use the Cholesky lower-triangular matrix to represent the variational
    # covariance. The following utility function extracts the lower-triangular
    # elements from its input list matrices.
    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVecsFromSRMatrices(
        srMatrices=qSigma0)

    # legendre quadrature points and weights
    trials_start_times = [trial_start_time for r in range(nTrials)]
    trials_end_times = [trial_end_time for r in range(nTrials)]
    legQuadPoints, legQuadWeights = \
        utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(
            nQuad=nQuad, trials_start_times=trials_start_times,
            trials_end_times=trials_end_times)

    # variational initial parameters
    qUParams0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}

    # kernel matrices initial parameters
    kmsParams0 = {"kernelsParams0": kernelsScaledParams0,
                  "inducingPointsLocs0": Z0}

    # psterior on inducing points initial parameters
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}

    # embedding initial parameters
    qHParams0 = {"C0": C0, "d0": d0}

    # all initial paramters
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}

    # quadrature initial parameters
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    # create model
    kernelMatrixInvMethod = stats.svGPFA.svGPFAModelFactory.kernelMatrixInvChol
    indPointsCovRep = stats.svGPFA.svGPFAModelFactory.indPointsCovChol
    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModelPyTorch(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels, kernelMatrixInvMethod=kernelMatrixInvMethod,
        indPointsCovRep=indPointsCovRep)

    model.setInitialParamsAndData(
        measurements=spikesTimes,
        initialParams=initialParams,
        eLLCalculationParams=quadParams,
        indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)

    # set EM optimization parameters
    optimMethod = "EM"
    optimParams = dict(
        em_max_iter=30,
        #
        estep_estimate=True,
        estep_optim_params=dict(
            max_iter=20,
            lr=1.0,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe"
        ),
        #
        mstep_embedding_estimate=True,
        mstep_embedding_optim_params=dict(
            max_iter=20,
            lr=1.0,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe"
        ),
        #
        mstep_kernels_estimate=True,
        mstep_kernels_optim_params=dict(
            max_iter=20,
            lr=1.0,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe"
        ),
        #
        mstep_indpointslocs_estimate=True,
        mstep_indpointslocs_optim_params=dict(
            max_iter=20,
            lr=1.0,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe"
        ),
        verbose=True
    )

    # maximize lower bound
    svEM = stats.svGPFA.svEM.SVEM_PyTorch()
    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
        svEM.maximize(model=model, optimParams=optimParams, method=optimMethod)

    pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
