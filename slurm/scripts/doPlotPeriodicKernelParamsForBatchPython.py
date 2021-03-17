
import sys
import os
import pdb
import pickle
import argparse
import csv
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"
sys.path.append("../../src")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptorsFilename", help="Name of file containing the descriptor of each line in the data files", default="../data/descriptors_20secSim.csv")
    parser.add_argument("--labelsAndEstNumbersFilename", help="Filename containing the labels ane model estimation numbers of all models to plot",
                        default="../data/labelsEstNumbers_20secSim_PyTorch_LBFGS_MAP.csv")
    parser.add_argument("--modelFilenamePattern", help="Filename of the pickle file where the model was saved", default="../../scripts/{:s}")
    parser.add_argument("--generativePeriod", help="Generative period value", type=float, default=5.0)
    parser.add_argument("--generativeLengthscale", help="Generative lengthscale value", type=float, default=2.25)
    parser.add_argument("--generativeParamsLineDash", help="Line dash for generative params", default="dash")
    parser.add_argument("--figFilenamePattern", help="Filename pattern of the plot figure", default="figures/periodicKernelParamsOfModelBatch_Pytorch_LBFGS_MAP.{:s}")
    parser.add_argument("--xlabel", help="Figure xlabel", default="Period")
    parser.add_argument("--ylabel", help="Figure ylabel", default="Lengthscale")
    args = parser.parse_args()
    descriptorsFilename = args.descriptorsFilename
    labelsAndEstNumbersFilename = args.labelsAndEstNumbersFilename
    modelFilenamePattern = args.modelFilenamePattern
    generativePeriod = float(args.generativePeriod)
    generativeLengthscale = float(args.generativeLengthscale)
    generativeParamsLineDash = args.generativeParamsLineDash
    figFilenamePattern = args.figFilenamePattern
    xlabel = args.xlabel
    ylabel = args.ylabel

    descriptors = pd.read_csv(descriptorsFilename, sep=" ", header=None).iloc[:,0].tolist()
    fig = go.Figure()
    with open(labelsAndEstNumbersFilename) as f:
        csvReader = csv.reader(f, delimiter=" ")
        i = 0
        for row in csvReader:
            label = descriptors[i]
            partialModelFilename = row[1]
            if len(partialModelFilename)>0:
                modelFilename = modelFilenamePattern.format(partialModelFilename)
                with open(modelFilename, "rb") as f: res = pickle.load(f)
                iterationModelParams = res["iterationModelParams"]
                trace = go.Scatter(
                    x=iterationModelParams[:,1], # period
                    y=iterationModelParams[:,0], # lengthscale
                    mode="lines+markers",
                    name=label,
                    showlegend=True,
                )
                fig.add_trace(trace)
            i += 1
    fig.add_vline(x=generativePeriod, line_dash=generativeParamsLineDash)
    fig.add_hline(y=generativeLengthscale, line_dash=generativeParamsLineDash)
    fig.update_yaxes(title_text=ylabel)
    fig.update_xaxes(title_text=xlabel)

    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))

    # fig.show()

if __name__=="__main__":
    main(sys.argv)
