import sys
import os
import pdb
import random
import scipy.io
import numpy as np
import torch
import pickle
import argparse
import configparser

sys.path.append("../src")
import stats.svGPFA.svGPFAModelFactory
import stats.svGPFA.svEM
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import utils.svGPFA.initUtils

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("mEstNumber", help="Matlab's estimation number", type=int)
    args = parser.parse_args()

    mEstNumber = args.mEstNumber

    mEstParamsFilename = "../../matlabCode/scripts/results/{:08d}-pointProcessEstimationParams.ini".format(mEstNumber)

    mEstConfig = configparser.ConfigParser()
    mEstConfig.read(mEstParamsFilename)
    pEstResNumber = int(mEstConfig["data"]["pEstNumber"])

    estimationDataForMatlabFilename = "results/{:08d}_estimationDataForMatlab.mat".format(pEstResNumber)
    pEstimMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(pEstResNumber)
    pEstConfig = configparser.ConfigParser()
    pEstConfig.read(pEstimMetaDataFilename)
    pSimNumber = int(pEstConfig["simulation_params"]["simResNumber"])

    pSimResMetaDataFilename = "results/{:08d}_simulation_metaData.ini".format(pSimNumber)
    pSimResMetaDataConfig = configparser.ConfigParser()
    pSimResMetaDataConfig.read(pSimResMetaDataFilename)
    pSimInitConfigFilename = pSimResMetaDataConfig["simulation_params"]["simInitConfigFilename"]
    pSimResFilename = pSimResMetaDataConfig["simulation_results"]["simResFilename"]

    with open(pSimResFilename, "rb") as f: simRes = pickle.load(f)
    latentsTrialsTimes = simRes["times"]

    loadRes = scipy.io.loadmat(estimationDataForMatlabFilename)
    loadRes.update({"latentsTrialsTimes_{:d}".format(0): latentsTrialsTimes[0].numpy().astype(np.float64)})
    scipy.io.savemat(file_name=estimationDataForMatlabFilename, mdict=loadRes)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
