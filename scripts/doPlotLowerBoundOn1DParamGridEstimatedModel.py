
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

def plotLowerBounds(tLowerBoundValues, eLowerBoundValues, xlabel, yMin, yMax, tColor="blue", eColor="red"):
    layout = {
        "xaxis": {"title": xlabel},
        # "yaxis": {"title": "Lower Bound"},
        "yaxis": {"title": "Lower Bound", "range": [yMin, yMax]},
    }
    data = []
    data.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "color": tColor,
                "x": tParamValues,
                "y": tLowerBoundValues,
            },
    )
    fig = go.Figure(
        data=data,
        layout=layout,
    )
    fig.add_shape(
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
    pio.renderers.default = "browser"

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estNumber", help="estimation result number", type=int)
    parser.add_argument("--nQuad", help="Number of quadrature points", type=int, default=200)
    args = parser.parse_args()
    estNumber = args.estNumber
    nQuad = args.nQuad

    estMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(estNumber)
    modelFilename = "results/{:08d}_estimatedModel.pickle".format(estNumber)

    estMetaDataConfig = configparser.ConfigParser()
    estMetaDataConfig.read(estMetaDataFilename)
    simResNumber = int(estMetaDataConfig["simulation_params"]["simResNumber"])

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
    indPointsLocsKMSRegEpsilon = float(simInitConfig["control_variables"]["indPointsLocsKMSRegEpsilon"])

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

    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=False)
    kernelsParams0 = utils.svGPFA.initUtils.getKernelsParams0(kernels=kernels, noiseSTD=0.0)

    Z0 = utils.svGPFA.configUtils.getIndPointsLocs0(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
    qUParams0 = {"qMu0": qMu0, "qSRSigma0": KzzChol}
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
    with open(modelFilename, "rb") as f: modelRes = pickle.load(f)
    model = modelRes["model"]

    paramValueStart = 1.0
    paramValueEnd = 20
    paramValueStep = .01
    yMax = None
    yMin = None
    paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
    lowerBoundValues = np.empty(paramValues.shape)
    kernelsParams = model.getKernelsParams()
    trueParam = kernelsParams[0][0].item()
    paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
    lowerBoundValues = np.empty(paramValues.shape)
    for i in range(len(paramValues)):
        kernelsParams[0][0] = paramValues[i]
        model.buildKernelsMatrices()
        lowerBoundValues[i] = model.eval()
    xlabel = "Lengthscale Value"

#     paramValueStart = 1.0
#     paramValueEnd = 20
#     paramValueStep = .01
#     yMax = 300
#     yMin = -5000
#     paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
#     lowerBoundValues = np.empty(paramValues.shape)
#     kernelsParams = model.getKernelsParams()
#     trueParam = kernelsParams[0][1].item()
#     paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
#     lowerBoundValues = np.empty(paramValues.shape)
#     for i in range(len(paramValues)):
#         kernelsParams[0][1] = paramValues[i]
#         model.buildKernelsMatrices()
#         lowerBoundValues[i] = model.eval()
#     xlabel = "Period Value"

#     embeddingParams = model.getSVEmbeddingParams()
#     trueParam = embeddingParams[0][0,0].item()
#     paramValueStart = -4
#     paramValueEnd = 4
#     paramValueStep = .01
#     yMax = 300
#     yMin = -5000
#     paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
#     lowerBoundValues = np.empty(paramValues.shape)
#     for i in range(len(paramValues)):
#         embeddingParams[0][0,0] = paramValues[i]
#         lowerBoundValues[i] = model.eval()
#     xlabel = "C[0,0] Value"

#     tEmbeddingParams = model.getSVEmbeddingParams()
#     tParamValue = tEmbeddingParams[1][0].item()
#     eEmbeddingParams = eModel.getSVEmbeddingParams()
#     eParamValue = eEmbeddingParams[1][0].item()
#     paramValueStart = -4
#     paramValueEnd = 4
#     paramValueStep = .01
#     yMax = 300
#     yMin = -700
#     paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
#     eLowerBoundValues = np.empty(paramValues.shape)
#     for i in range(len(paramValues)):
#         tEmbeddingParams[1][0] = paramValues[i]
#         tLowerBoundValues[i] = model.eval()
#         eEmbeddingParams[1][0] = paramValues[i]
#         eLowerBoundValues[i] = eModel.eval()
#     tEmbeddingParams[1][0] = tParamValue
#     eEmbeddingParams[1][0] = eParamValue

#     plotLowerBounds(tLowerBoundValues=tLowerBoundValues, eLowerBoundValues=eLowerBoundValues, xlabel="d[0] Value")
#     fig.show()
#     pdb.set_trace()

    layout = {
        "xaxis": {"title": xlabel},
        # "yaxis": {"title": "Lower Bound"},
        "yaxis": {"title": "Lower Bound", "range": [yMin, yMax]},
    }
    data = []
    data.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": paramValues,
                "y": lowerBoundValues,
            },
    )
    fig = go.Figure(
        data=data,
        layout=layout,
    )
    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            x0=trueParam,
            y0=yMin,
            x1=trueParam,
            y1=yMax,
            line=dict(
                color="Blue",
                width=3
            )
    ))
    pio.renderers.default = "browser"
    fig.show()
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

