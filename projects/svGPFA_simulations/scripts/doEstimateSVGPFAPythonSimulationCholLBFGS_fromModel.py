import sys
import os
import pdb
import random
import torch
import pickle
import argparse
import configparser

sys.path.append("../src")
import stats.svGPFA.svGPFAModelFactory
import stats.svGPFA.svLBFGS
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import utils.svGPFA.initUtils

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("estInitNumber", help="estimation init number", type=int)
    parser.add_argument("initEstResNumber", help="initial model estimation number", type=int)
    args = parser.parse_args()

    estInitNumber = args.estInitNumber
    initEstResNumber = args.initEstResNumber

    estInitConfigFilename = "data/{:08d}_estimation_metaData.ini".format(estInitNumber)
    estInitConfig = configparser.ConfigParser()
    estInitConfig.read(estInitConfigFilename)

    optimParamsConfig = estInitConfig._sections["optim_params"]
    optimMethod = optimParamsConfig["em_method"]
    optimParams = {}
    optimParams["em_max_iter"] = int(optimParamsConfig["em_max_iter"])
    optimParams["verbose"] = optimParamsConfig["verbose"]=="True"
    optimParams["LBFGS_optim_params"] = {
        "max_iter": int(optimParamsConfig["max_iter"]),
        "lr": float(optimParamsConfig["lr"]),
        "tolerance_grad": float(optimParamsConfig["tolerance_grad"]),
        "tolerance_change": float(optimParamsConfig["tolerance_change"]),
        "line_search_fn": optimParamsConfig["line_search_fn"],
    }

    # choose modelSaveFilename
    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estimResMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(estResNumber)
        if not os.path.exists(estimResMetaDataFilename):
           estPrefixUsed = False
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)

    # load initial model
    initModelFilename = "results/{:08d}_estimatedModel.pickle".format(initEstResNumber)
    with open(initModelFilename, "rb") as f: modelRes = pickle.load(f)
    model = modelRes["model"]

    # maximize lower bound
    svLBFGS = stats.svGPFA.svLBFGS.SVLBFGS()
    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams  = svLBFGS.maximize(model=model, optimParams=optimParams)

    initResMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(initEstResNumber)
    initResConfig = configparser.ConfigParser()
    initResConfig.read(initResMetaDataFilename)
    simResNumber = int(initResConfig["simulation_params"]["simResNumber"])

    # save estimated values
    estimResConfig = configparser.ConfigParser()
    estimResConfig["initial_model"] = {"initEstResNumber": initEstResNumber}
    estimResConfig["simulation_params"] = {"simResNumber": simResNumber}
    estimResConfig["optim_params"] = optimParams
    estimResConfig["estimation_params"] = {"estInitNumber": estInitNumber}
    with open(estimResMetaDataFilename, "w") as f: estimResConfig.write(f)

    resultsToSave = {"lowerBoundHist": lowerBoundHist, "elapsedTimeHist": elapsedTimeHist, "terminationInfo": terminationInfo, "iterationModelParams": iterationsModelParams, "model": model}
    with open(modelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)
    print("Saved results to {:s}".format(modelSaveFilename))

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
