
import sys
import pdb
import math
import numbers
import pickle
import pandas as pd
import numpy as np
import scipy.io
import argparse
import plotly.graph_objs as go
sys.path.append("../../src")

def getPeriodParamsAtConvergence(modelFilenames, periodScale):
    periodParamsAtConvergence = [None]*len(modelFilenames)
    for i, modelFilename in enumerate(modelFilenames):
        if not isinstance(modelFilename, numbers.Number):
            with open(modelFilename, "rb") as f:
                loadRes = pickle.load(f)
            periodParam = loadRes["model"].getKernelsParams()[0][1]/periodScale
            periodParamsAtConvergence[i] = periodParam
        else:
            periodParamsAtConvergence[i] = float("nan")
    return periodParamsAtConvergence

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptorsFilename", help="Name of file containing the descriptor of each line in data files", default="../data/descriptors_20secSim.csv")
    parser.add_argument("--labelsAndEstNumbersFilename", help="Filename containing the labels and model estimation numbers to plot", default="../data/labelsEstNumbers_20secSim_SciPy_L-BFGS-B_MAP.csv")
    parser.add_argument("--generativePeriod", help="Value of othe generative period parameter", type=float, default=5.0)
    parser.add_argument("--periodScale", help="Scale for period parameter", type=float, default=1.0)
    parser.add_argument("--modelFilenamePattern", help="Filename pattern for a models", default="../../scripts/{:s}")
    parser.add_argument("--generativePeriodLineDash", help="Line dash for generative period", default="dash")
    parser.add_argument("--xlab", help="Figure label for the abcissa", default="Period Parameter Initial Condition")
    parser.add_argument("--ylab", help="Figure label for the ordinate", default="EStimated Period")
    parser.add_argument("--figFilenamePattern", help="Figure filename pattern", default="../figures/20secSim_SciPy_L-BFGS-B_estimatedVsInitialperiod.{:s}")
    args = parser.parse_args()

    descriptorsFilename = args.descriptorsFilename
    labelsAndEstNumbersFilename = args.labelsAndEstNumbersFilename
    generativePeriod = args.generativePeriod
    periodScale = args.periodScale
    modelFilenamePattern = args.modelFilenamePattern
    generativePeriodLineDash = args.generativePeriodLineDash
    xlab = args.xlab
    ylab = args.ylab
    figFilenamePattern = args.figFilenamePattern

    partialModelsFilenames = pd.read_csv(labelsAndEstNumbersFilename, sep=" ", header=0).iloc[:,1].tolist()
    fullModelsFilenames = []
    for partialModelFilename in partialModelsFilenames:
        if not isinstance(partialModelFilename, numbers.Number):
            fullModelsFilenames.append(modelFilenamePattern.format(partialModelFilename))
        else:
            fullModelsFilenames.append(partialModelFilename)
    values = getPythonPeriodParamsAtConvergence(modelFilenames=fullModelsFilenames, periodScale=periodScale)

    descriptors = pd.read_csv(descriptorsFilename, sep=" ").iloc[:,0].tolist()

    trace = go.Scatter(
        x = descriptors,
        y = values,
        mode="lines+markers",
        name="SciPy L-BFGS-B MAP",
        showlegend=True,
    )

    fig = go.Figure()
    fig.add_trace(trace)
    fig.add_hline(y=generativePeriod, line_dash=generativePeriodLineDash)
    fig.update_xaxes(title_text=xlab)
    fig.update_yaxes(title_text=ylab)
    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
