
import sys
import os
import pdb
import pickle
import argparse
import csv
import numpy as np
import plotly.graph_objs as go
sys.path.append("../src")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelsAndEstNumbersFilename", help="Filename containing the labels ane model estimation numbers of all models to plot", default="notes/labelsEstNumbers.csv")
    parser.add_argument("--modelFilenamePattern", help="Filename of the pickle file where the model was saved", default="results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--figFilenamePattern", help="Filename pattern of the plot figure", default="figures/lowerBoundsVs{:s}OfModelBatch.{:s}")
    parser.add_argument("--xlabelIterationNumber", help="Iteration number figure xlabel", default="Iteration Number")
    parser.add_argument("--xlabelElapsedTime", help="Elapsed time igure xlabel", default="Elapsed Time (sec)")
    parser.add_argument("--ylabel", help="Figure ylabel", default="Lower Bound")
    args = parser.parse_args()
    labelsAndEstNumbersFilename = args.labelsAndEstNumbersFilename
    modelFilenamePattern = args.modelFilenamePattern
    figFilenamePattern = args.figFilenamePattern
    xlabelIterationNumber = args.xlabelIterationNumber
    xlabelElapsedTime = args.xlabelElapsedTime
    ylabel = args.ylabel

    figIteration = go.Figure()
    figElapsedTime = go.Figure()
    with open(labelsAndEstNumbersFilename) as f:
        csvReader = csv.reader(f, delimiter=" ")
        for row in csvReader:
            label = row[0]
            estNumber = int(row[1])
            modelFilename = modelFilenamePattern.format(estNumber)
            with open(modelFilename, "rb") as f: res = pickle.load(f)
            lowerBoundHist = res["lowerBoundHist"]
            traceIteration = go.Scatter(
                x=np.arange(len(lowerBoundHist)),
                y=lowerBoundHist,
                mode="lines",
                name=label,
                showlegend=True,
            )
            figIteration.add_trace(traceIteration)
            elapsedTimeHist = res["elapsedTimeHist"]
            traceElapsedTime = go.Scatter(
                x=elapsedTimeHist,
                y=lowerBoundHist,
                mode="lines",
                name=label,
                showlegend=True,
            )
            figElapsedTime.add_trace(traceElapsedTime)
    figIteration.update_yaxes(title_text=ylabel)
    figIteration.update_xaxes(title_text=xlabelIterationNumber)
    figElapsedTime.update_yaxes(title_text=ylabel)
    figElapsedTime.update_xaxes(title_text=xlabelElapsedTime)

    figIteration.write_image(figFilenamePattern.format("Iteration", "png"))
    figIteration.write_html(figFilenamePattern.format("Iteration", "html"))

    figElapsedTime.write_image(figFilenamePattern.format("ElapsedTime", "png"))
    figElapsedTime.write_html(figFilenamePattern.format("ElapsedTime", "html"))

    figIteration.show()
    figElapsedTime.show()

if __name__=="__main__":
    main(sys.argv)
