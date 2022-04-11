
import sys
import os
import pdb
import pickle
import argparse
import csv
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"
sys.path.append("../src")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelsAndEstNumbersFilename", help="Filename containing the labels ane model estimation numbers of all models to plot",
                        default="../slurm/data/labelsEstNumbers20secSim.csv")
    parser.add_argument("--modelFilenamePattern", help="Filename of the pickle file where the model was saved", default="results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--figFilenamePattern", help="Filename pattern of the plot figure", default="figures/periodicKernelParamsOfModelBatch.{:s}")
    parser.add_argument("--xlabel", help="Figure xlabel", default="Period")
    parser.add_argument("--ylabel", help="Figure ylabel", default="Lengthscale")
    args = parser.parse_args()
    labelsAndEstNumbersFilename = args.labelsAndEstNumbersFilename
    modelFilenamePattern = args.modelFilenamePattern
    figFilenamePattern = args.figFilenamePattern
    xlabel = args.xlabel
    ylabel = args.ylabel

    fig = go.Figure()
    with open(labelsAndEstNumbersFilename) as f:
        csvReader = csv.reader(f, delimiter=" ")
        for row in csvReader:
            label = row[0]
            modelFilename = row[1]
            with open(modelFilename, "rb") as f: res = pickle.load(f)
            iterationModelParams = res["iterationModelParams"]
            trace = go.Scatter(
                x=iterationModelParams[:,1], # period
                y=iterationModelParams[:,0], # lengthscale
                # mode="lines+markers",
                mode="markers",
                name=label,
                showlegend=True,
            )
            fig.add_trace(trace)
    fig.update_yaxes(title_text=ylabel)
    fig.update_xaxes(title_text=xlabel)

    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))

    fig.show()

if __name__=="__main__":
    main(sys.argv)
