
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
    parser.add_argument("estNumber", help="estimation result number", type=int)
    parser.add_argument("--paramValueStart", help="Start parameter value", type=float, default=2.7)
    parser.add_argument("--paramValueEnd", help="End parameters value", type=float, default=20.0)
    parser.add_argument("--paramValueStep", help="Step for parameter values", type=float, default=0.01)
    parser.add_argument("--yMin", help="Minimum y value", type=float, default=-math.inf)
    parser.add_argument("--yMax", help="Minimum y value", type=float, default=+math.inf)
    parser.add_argument("--nQuad", help="Number of quadrature points", type=int, default=200)
    args = parser.parse_args()
    estNumber = args.estNumber
    paramValueStart = args.paramValueStart
    paramValueEnd = args.paramValueEnd
    paramValueStep = args.paramValueStep
    yMin = args.yMin
    yMax = args.yMax
    nQuad = args.nQuad

    modelFilename = "results/{:08d}_estimatedModel.pickle".format(estNumber)
    with open(modelFilename, "rb") as f: modelRes = pickle.load(f)
    model = modelRes["model"]

    paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
    lowerBoundValues = np.empty(paramValues.shape)
    for i in range(len(paramValues)):
        kernelsParams = model.getKernelsParams()
        kernelsParams[0][1] = paramValues[i]
        model.buildKernelsMatrices()
        lowerBoundValues[i] = model.eval()
    xlabel = "Period Value"

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
    pio.renderers.default = "browser"
    fig.show()
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

