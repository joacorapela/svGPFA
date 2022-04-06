import sys
import time
import torch
import pickle

sys.path.append("../src")
import stats.kernels
import stats.svGPFA.svGPFAModelFactory
import stats.svGPFA.svEM
import stats.pointProcess.tests
import utils.svGPFA.miscUtils
import utils.svGPFA.initUtils
import plot.svGPFA.plotUtilsPlotly

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
ksTest_gamma = 20                                 # number of simulations for the KS test numerical correction

# load spikesTimes
# spikesTimes should be a list of lists
# spikesTimes[r][n] is the list of spikes times or neuron n in trial r
with open(simResFilename, "rb") as f:
    simRes = pickle.load(f)
spikesTimes = simRes["spikes"]

# embedding parameters initial values: uniform[0,1]
# Duncker and Sahani, 2018, Eq. 1 (middle)
C0 = torch.normal(mean=0.0, std=1.0, size=(nNeurons, nLatents), dtype=torch.double).contiguous()
d0 = torch.normal(mean=0.0, std=1.0, size=(nNeurons, 1), dtype=torch.double).contiguous()

# kernels of latents: all ExponentialQuadratic Kernels
# Duncker and Sahani, 2018, Eq. 1 (top)
kernels = [[] for r in range(nLatents)]
for k in range(nLatents):
    kernels[k] = stats.kernels.ExponentialQuadraticKernel()

# kernels parameters initial values: all kernels have the same initial
# lengthscale0
# Duncker and Sahani, 2018, Eq. 1 (top)
kernelsScaledParams0 = [torch.tensor([lengthscale0], dtype=torch.double)
                        for r in range(nLatents)]

# inducing points locations initial values: equally spaced nIndPoints
# between trial_start_time and trial_end_time
# Duncker and Sahani, 2018, paragraph above Eq. 2
Z0 = [[] for k in range(nLatents)]
for k in range(nLatents):
    Z0[k] = torch.empty((nTrials, nIndPoints, 1), dtype=torch.double)
    for r in range(nTrials):
        Z0[k][r, :, 0] = torch.linspace(trial_start_time, trial_end_time,
                                        nIndPoints, dtype=torch.double)

# variational mean initial value: Uniform[0, 1]
# Duncker and Sahani, 2018, m_k in paragraph above Eq. 4
qMu0 = [[] for r in range(nLatents)]
for k in range(nLatents):
    qMu0[k] = torch.empty((nTrials, nIndPoints, 1), dtype=torch.double)
    for r in range(nTrials):
        qMu0[k][r, :, 0] = torch.normal(mean=0.0, std=1.0, size=(nIndPoints, 1))

# variational covariance initial value: Identity*1e-2
# Duncker and Sahani, 2018, V_k in paragraph above Eq. 4
diag_value = 1e-2
qSigma0 = [[] for r in range(nLatents)]
for k in range(nLatents):
    qSigma0[k] = torch.empty((nTrials, nIndPoints, nIndPoints),
                             dtype=torch.double)
    for r in range(nTrials):
        qSigma0[k][r, :, :] = torch.eye(nIndPoints)*diag_value

# we use the Cholesky lower-triangular matrix to represent the variational
# covariance. The following utility function extracts the lower-triangular
# elements from its input of list matrices.
srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVecsFromSRMatrices(
    srMatrices=qSigma0)

# legendre quadrature points and weights used to calculate the integral in
# the first term of Eq. 7 in Duncker and Sahani, 2018.
trials_start_times = [trial_start_time for r in range(nTrials)]
trials_end_times = [trial_end_time for r in range(nTrials)]
legQuadPoints, legQuadWeights = \
    utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(
        nQuad=nQuad, trials_start_times=trials_start_times,
        trials_end_times=trials_end_times)

# Finally, we build the dictionaries of initial parameters and quadrature
# parameters used to initialize the svGPFA model
qUParams0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
kmsParams0 = {"kernelsParams0": kernelsScaledParams0,
              "inducingPointsLocs0": Z0}
qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
             "kernelsMatricesStore": kmsParams0}
qHParams0 = {"C0": C0, "d0": d0}
initialParams = {"svPosteriorOnLatents": qKParams0,
                 "svEmbedding": qHParams0}
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
    # em_max_iter=30,
    em_max_iter=5,
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
tic = time.perf_counter()
lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
    svEM.maximize(model=model, optimParams=optimParams, method=optimMethod)
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")

resultsToSave = {"lowerBoundHist": lowerBoundHist,
                 "elapsedTimeHist": elapsedTimeHist,
                 "terminationInfo": terminationInfo,
                 "iterationModelParams": iterationsModelParams,
                 "model": model}
modelSaveFilename = "/tmp/model.pickle"
with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)
print("Saved results to {:s}".format(modelSaveFilename))

# Plotting
## imports for plotting
import numpy as np
import pandas as pd
import sklearn.metrics
import plot.svGPFA.plotUtilsPlotly

## Set neuron, latent and trial to plot and compute trial_times
neuronToPlot = 0
trialToPlot = 0
latentToPlot = 0
# first define the times at wich latents log pre-intensities and CIFs will be
# calculated
sampling_rate = 100.0 # Hz
trial_times = torch.arange(trial_start_time, trial_end_time, 1.0/sampling_rate)

## plot lower bound history
fig = plot.svGPFA.plotUtilsPlotly.getPlotLowerBoundHist(lowerBoundHist=
                                                        lowerBoundHist)
fig.show()

## KS test time rescaling with numerical correction
trialForGOF = 0
neuronForGOF = 0
with torch.no_grad():
    epmcifValues = model.computeExpectedPosteriorCIFs(times=trial_times)
cifValuesKS = epmcifValues[trialForGOF][neuronForGOF]
spikesTimesKS = spikesTimes[trialForGOF][neuronForGOF]
diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = stats.pointProcess.tests.KSTestTimeRescalingNumericalCorrection(spikesTimes=spikesTimesKS, cifTimes=trial_times, cifValues=cifValuesKS, gamma=ksTest_gamma)
title = "Trial {:d}, Neuron {:d}".format(trialForGOF, neuronForGOF)
fig = plot.svGPFA.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx, estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb, title=title)
fig.show()

## ROC predictive analysis
dt = (trial_times[1] - trial_times[0]).item()
pk = cifValuesKS.detach().numpy() * dt
bins = pd.interval_range(start=trial_start_time, end=trial_end_time, periods=len(pk))
cutRes, _ = pd.cut(spikesTimesKS.tolist(), bins=bins, retbins=True)
Y = cutRes.value_counts().values
fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, pk, pos_label=1)
roc_auc = sklearn.metrics.auc(fpr, tpr)
title = "Trial {:d}, Neuron {:d}".format(trialForGOF, neuronForGOF)
fig = plot.svGPFA.plotUtilsPlotly.getPlotResROCAnalysis(fpr=fpr, tpr=tpr, auc=roc_auc, title=title)
fig.show()

# get model parameters

# kernels_params is a list of length nLatents
# kernels_params[r] contains the kernels parameter for latent r
# for this example kernel_params[r] contains the estimated lengthscale
# corresponding to the latent r
kernels_params = model.getKernelsParams()

# embedding_params is a list of length 2
# embedding_params[0] is the embedding matrix C; Eq. 1, middle, D&S18
# embedding_params[1] is the embedding vector d; Eq. 1, middle, D&S18
embedding_params = model.getSVEmbeddingParams()

# indPoints_params is a list of length 2*nLatents
indPoints_params = model.getSVPosteriorOnIndPointsParams()
# the first nLatents elements contain the inducing points mean
indPoints_means = indPoints_params[:nLatents]
# the last nLatents elements contain the Cholesky representation of the
# inducing points covariances
srQSigmaVecs = indPoints_params[nLatents:]
# the following function builds the inducing points covariances from their
# Cholesky representations
indPoints_covs = utils.svGPFA.miscUtils.buildQSigmasFromSRQSigmaVecs(srQSigmaVecs=srQSigmaVecs)

# indPointsLocs is a list of length nLatents
# indPointsLocs[k] \in nTrials x nIndPoints x 1
# indPointsLocs[k][r, :, 0] gives the inducing points for latent k and trial r
indPointsLocs = model.getIndPointsLocs()

# get latents, log pre-intensities and CIFs

# latents is a list of length nLatents
# latents[k] \in nTrials x nTimes, 2
# latents[k][r, :, 0] gives the kth latent mean for trial r at trial_times
# latents[k][r, :, 0] gives the kth latent variance for trial r at trial_times
latents = model.predictLatents(times=trial_times)

# embedding is a list of length 2
# embedding[0] \in nTrials x nTimes x nNeurons
# embedding[0][r, :, n] gives the embedding mean for trial r and neuron n
# embedding[1] \in nTrials x nTimes x nNeurons
# embedding[1][r, :, n] gives the embedding variance for trial r and neuron n
embedding = model.predictEmbedding(times=trial_times)

# cifs is a list of lists
# cif[r][n] \in nTimes gives the CIF of neuron n in trial r
cifs = model.computeExpectedPosteriorCIFs(times=trial_times)

