
import sys
import os
import torch
import pdb
import pickle
import argparse
import configparser
from scipy.io import loadmat
import plotly.io as pio
sys.path.append(os.path.expanduser("../src"))
import plot.svGPFA.plotUtilsPlotly
import utils.svGPFA.configUtils
import utils.svGPFA.initUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("mEstNumber", help="Matlab's estimation number", type=int)
    args = parser.parse_args()
    mEstNumber = args.mEstNumber

    mEstParamsFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessEstimationParams.ini".format(mEstNumber)
    mEstConfig = configparser.ConfigParser()
    mEstConfig.read(mEstParamsFilename)
    pEstNumber = int(mEstConfig["data"]["pEstNumber"])

    pEstimParamsFilename = "results/{:08d}_estimation_metaData.ini".format(pEstNumber)
    pEstConfig = configparser.ConfigParser()
    pEstConfig.read(pEstimParamsFilename)
    pSimNumber = int(pEstConfig["simulation_params"]["simResNumber"])

    pSimResMetaDataFilename = "results/{:08d}_simulation_metaData.ini".format(pSimNumber)
    pSimResMetaDataConfig = configparser.ConfigParser()
    pSimResMetaDataConfig.read(pSimResMetaDataFilename)
    pSimInitConfigFilename = pSimResMetaDataConfig["simulation_params"]["simInitConfigFilename"]

    pSimInitConfig = configparser.ConfigParser()
    pSimInitConfig.read(pSimInitConfigFilename)
    nIndPointsPerLatent = [int(str) for str in pSimInitConfig["control_variables"]["nIndPointsPerLatent"][1:-1].split(",")]
    nLatents = len(nIndPointsPerLatent)
    tKernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=pSimInitConfig, forceUnitScale=True)
    kernelsTypes = [type(tKernels[k]).__name__ for k in range(nLatents)]
    tKernelsParams = utils.svGPFA.initUtils.getKernelsParams0(kernels=tKernels, noiseSTD=0.0)

    mModelSaveFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat".format(mEstNumber)
    pModelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(pEstNumber)
    figFilenamePattern = "figures/{:08d}-{:08d}-truePythonMatlabKernelsParamsPointProcess.{{:s}}".format(mEstNumber, pEstNumber)

    with open(pModelSaveFilename, "rb") as f: res = pickle.load(f)
    pModel = res["model"]
    pKernelsParams = pModel.getKernelsParams()

    loadRes = loadmat(mModelSaveFilename)
    mKernelsParams = [[] for k in range(nLatents)]
    for k in range(nLatents):
        mKernelsParams[k] = loadRes["m"][0,0]["kerns"][k,0]["hprs"][0,0].squeeze()

    fig = plot.svGPFA.plotUtilsPlotly.getPlotTruePythonAndMatlabKernelsParams(
        kernelsTypes=kernelsTypes,
        trueKernelsParams=tKernelsParams,
        pythonKernelsParams=pKernelsParams,
        matlabKernelsParams=mKernelsParams)
    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))
    pio.renderers.default = "browser"
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
