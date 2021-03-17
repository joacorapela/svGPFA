
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

def getLowerBoundsAtConvergence(modelFilenames, convergenceTolerance):
    lowerBoundsAtConvergence = [None]*len(modelFilenames)
    for i, modelFilename in enumerate(modelFilenames):
        if not isinstance(modelFilename, numbers.Number):
            with open(modelFilename, "rb") as f:
                loadRes = pickle.load(f)
            lowerBoundHist = np.array(loadRes["lowerBoundHist"])
            lowerBoundsDiff = lowerBoundHist[1:]-lowerBoundHist[:-1]
            indices = np.where(lowerBoundsDiff<convergenceTolerance)[0]
            if len(indices)>0:
                lowerBoundsAtConvergence[i] = lowerBoundHist[indices[0]]
            else:
                lowerBoundsAtConvergence[i] = lowerBoundHist[-1]
        else:
                lowerBoundsAtConvergence[i] = float("nan")
    return lowerBoundsAtConvergence

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptorsFilename", help="Name of file containing the descriptor of each line in the data files", default="../data/descriptors_20secSim.csv")
    parser.add_argument("--labelsAndEstNumbersFilename", help="Filename containing the labels and model estimation numbers to plot", default="../data/labelsEstNumbers_20secSim_SciPy_L-BFGS-B.csv")
    parser.add_argument("--modelFilenamePattern", help="Model filename pattern", default="../../scripts/{:s}")
    parser.add_argument("--convergenceTolerance", help="Lower boundd convergence tolerance", type=float, default=1e-5)
    parser.add_argument("--xlab", help="Figure label for the abcissa", default="Period Parameter Initial Condition")
    parser.add_argument("--ylab", help="Figure label for the ordinate", default="Lower Bound")
    parser.add_argument("--figFilenamePattern", help="Figure filename pattern", default="../figures/20secSim_lowerBounds_SciPy_L-BFGS-B_MAP.{:s}")
    args = parser.parse_args()

    descriptorsFilename = args.descriptorsFilename
    labelsAndEstNumbersFilename = args.labelsAndEstNumbersFilename
    modelFilenamePattern = args.modelFilenamePattern
    convergenceTolerance = args.convergenceTolerance
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
    values = getLowerBoundsAtConvergence(modelFilenames=fullModelsFilenames, convergenceTolerance=convergenceTolerance)

    descriptors = pd.read_csv(descriptorsFilename, sep=" ").iloc[:,0].tolist()

    trace = go.Scatter(
        x = descriptors,
        y = values,
        mode="lines+markers",
        name="SciPy L-BFGS-B",
        showlegend=True,
    )

    fig = go.Figure()
    fig.add_trace(trace)
    fig.update_xaxes(title_text=xlab)
    fig.update_yaxes(title_text=ylab)
    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
