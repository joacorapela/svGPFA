
import sys
import pdb
import math
import argparse
import scipy.io
import pickle
import configparser
import torch
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
sys.path.append("../src")
import utils.svGPFA.initUtils
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import stats.svGPFA.svGPFAModelFactory

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simResNumber", help="simulation result number", type=int)
    parser.add_argument("indPointsLocsKMSRegEpsilon", help="regularization epsilong for the inducing points locations covariance", type=float)
    parser.add_argument("--paramValueStart", help="Start parameter value", type=float, default=1.0)
    parser.add_argument("--paramValueEnd", help="End parameters value", type=float, default=20.0)
    parser.add_argument("--paramValueStep", help="Step for parameter values", type=float, default=0.01)
    parser.add_argument("--yMin", help="Minimum y value", type=float, default=-math.inf)
    parser.add_argument("--yMax", help="Minimum y value", type=float, default=+math.inf)
    parser.add_argument("--nQuad", help="Number of quadrature points", type=int, default=200)
    args = parser.parse_args()
    simResNumber = args.simResNumber
    indPointsLocsKMSRegEpsilon = args.indPointsLocsKMSRegEpsilon
    paramValueStart = args.paramValueStart
    paramValueEnd = args.paramValueEnd
    paramValueStep = args.paramValueStep
    yMin = args.yMin
    yMax = args.yMax
    nQuad = args.nQuad

    # load data and initial values
    simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]
    simResFilename = simResConfig["simulation_results"]["simResFilename"]

    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nIndPointsPerLatent = [int(str) for str in simInitConfig["control_variables"]["nIndPointsPerLatent"][1:-1].split(",")]
    nLatents = len(nIndPointsPerLatent)
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    firstIndPointLoc = float(simInitConfig["control_variables"]["firstIndPointLoc"])
    # indPointsLocsKMSRegEpsilon = float(simInitConfig["control_variables"]["indPointsLocsKMSRegEpsilon"])

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    KzzChol = simRes["KzzChol"]
    indPointsMeans = simRes["indPointsMeans"]
    C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(CFilename=simInitConfig["embedding_params"]["C_filename"], dFilename=simInitConfig["embedding_params"]["d_filename"])

    # patch to acommodate Lea's equal number of inducing points across trials
    qMu0 = [[] for k in range(nLatents)]
    for k in range(nLatents):
        qMu0[k] = torch.empty((nTrials, nIndPointsPerLatent[k], 1), dtype=torch.double)
        for r in range(nTrials):
            qMu0[k][r,:,:] = indPointsMeans[r][k]
    # end patch

    legQuadPoints, legQuadWeights = utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(nQuad=nQuad, trialsLengths=trialsLengths)

    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=True)
    kernelsParams0 = utils.svGPFA.initUtils.getKernelsParams0(kernels=kernels, noiseSTD=0.0)

    # Z0 = utils.svGPFA.initUtils.getIndPointLocs0(nIndPointsPerLatent=nIndPointsPerLatent, trialsLengths=trialsLengths, firstIndPointLoc=firstIndPointLoc)
    Z0 = utils.svGPFA.configUtils.getIndPointsLocs0(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)

    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVecsFromSRMatrices(srMatrices=KzzChol)
    qUParams0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C, "d0": d}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    # create model
    model = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels)

    model.setMeasurements(measurements=spikesTimes)
    model.setInitialParams(initialParams=initialParams)
    model.setQuadParams(quadParams=quadParams)
    pdb.set_trace()
    model.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)
    model.buildKernelsMatrices()
#     paramValueStart = 1.0
#     paramValueEnd = 20
#     paramValueStep = .01
#     yMax = None
#     yMin = None
#     paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
#     lowerBoundValues = np.empty(paramValues.shape)
#     kernelsParams = model.getKernelsParams()
#     trueParam = kernelsParams[0][1].item()
#     # begin debug
#     kernelsParams[0][1] = 4.45
#     # kernelsParams[0][1] = 5.0
#     model.buildKernelsMatrices()
#     # aux = model.eval()
#     # pdb.set_trace()
#     # end debug
#     for i in range(len(paramValues)):
#         kernelsParams[0][1] = paramValues[i]
#         model.buildKernelsMatrices()
#         lowerBoundValues[i] = model.eval()
#     xlabel = "Period Value"

    paramValueStart = 1.0
    paramValueEnd = 5.0
    paramValueStep = .01
    yMax = None
    yMin = None
    paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
    lowerBoundValues = np.empty(paramValues.shape)
    conditionNumbers = np.empty((nTrials, len(paramValues)))
    eValues = np.empty((nTrials, len(paramValues), 10, 2))
    kernelsParams = model.getKernelsParams()
    trueParam = kernelsParams[0][0].item()
    for i in range(len(paramValues)):
        kernelsParams[0][0] = paramValues[i]
        model.buildKernelsMatrices()
        lowerBoundValues[i] = model.eval()
        Kzz = model._eLL._svEmbeddingAllTimes._svPosteriorOnLatents._indPointsLocsKMS._Kzz
        for r in range(nTrials):
            eValues[r,i,:,:], _ = torch.eig(Kzz[0][r,:,:])
            conditionNumbers[r,i] = eValues[r,i,0,0]/eValues[r,i,-1,0]
    xlabel = "Lengthscale Value"

#     embeddingParams = model.getSVEmbeddingParams()
#     trueParam = embeddingParams[0][0,0].item()
#     paramValueStart = -4
#     paramValueEnd = 4
#     paramValueStep = .01
#     yMax = 0
#     yMin = -5000
#     paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
#     lowerBoundValues = np.empty(paramValues.shape)
#     for i in range(len(paramValues)):
#         embeddingParams[0][0,0] = paramValues[i]
#         lowerBoundValues[i] = model.eval()
#     xlabel = "C[0,0] Value"

#     embeddingParams = model.getSVEmbeddingParams()
#     trueParam = embeddingParams[1][0].item()
#     paramValueStart = -4
#     paramValueEnd = 4
#     paramValueStep = .01
#     yMax = math.inf
#     yMin = -math.inf
#     paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
#     lowerBoundValues = np.empty(paramValues.shape)
#     for i in range(len(paramValues)):
#         embeddingParams[1][0] = paramValues[i]
#         lowerBoundValues[i] = model.eval()
#     xlabel = "d[0] Value"

    layoutLowerBound = {
        "xaxis": {"title": xlabel},
        # "yaxis": {"title": "Lower Bound"},
        "yaxis": {"title": "Lower Bound", "range": [yMin, yMax]},
    }
    layoutConditionNumber = {
        "xaxis": {"title": xlabel},
        # "yaxis": {"title": "Lower Bound"},
        "yaxis": {"title": "Condition Number", "range": [yMin, yMax]},
    }
    layoutEvals = {
        "xaxis": {"title": xlabel},
        # "yaxis": {"title": "Lower Bound"},
        "yaxis": {"title": "Eigenvalue", "range": [yMin, yMax]},
    }
    dataLowerBound = []
    dataLowerBound.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": paramValues,
                "y": lowerBoundValues,
                "name": "lower bound",
            },
    )
    dataConditionNumber = []
    dataConditionNumber.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": paramValues,
                "y": conditionNumbers[0,:],
                "name": "cNum trial 0",
            },
    )
    dataConditionNumber.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": paramValues,
                "y": conditionNumbers[1,:],
                "name": "cNum trial 1",
            },
    )
    dataEvals = []
    for i in range(10):
        # eValues = np.empyt((nTrials, len(paramValues), 10, 2))
        dataEvals.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": paramValues,
                "y": eValues[0,:,i,0],
                "name": "real eigval {:d}, trial 0".format(i),
            },
        )
        dataEvals.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": paramValues,
                "y": eValues[0,:,i,1],
                "name": "imag eigval {:d}, trial 0".format(i),
            },
        )
        dataEvals.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": paramValues,
                "y": eValues[1,:,i,0],
                "name": "real eigval {:d}, trial 1".format(i),
            },
        )
        dataEvals.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": paramValues,
                "y": eValues[1,:,i,1],
                "name": "imag eigval {:d}, trial 1".format(i),
            },
        )
    figLowerBound = go.Figure(
        data=dataLowerBound,
        layout=layoutLowerBound,
    )
    figLowerBound.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=trueParam,
            y0=yMin,
            x1=trueParam,
            y1=yMax,
            line=dict(
                color="Red",
                width=3
            )
    ))
    figConditionNumber = go.Figure(
        data=dataConditionNumber,
        layout=layoutConditionNumber,
    )
    figEvals = go.Figure(
        data=dataEvals,
        layout=layoutEvals,
    )
    pio.renderers.default = "browser"
    figLowerBound.show()
    figConditionNumber.show()
    figEvals.show()
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

