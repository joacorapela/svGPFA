
import sys
import pdb
import argparse
import configparser
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("../src")
import utils.svGPFA.configUtils
import simulations.svGPFA.utils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simInitConfigNumber", help="Simulation initialization configuration number", type=int)
    parser.add_argument("latentDescriptor", help="Latent descriptor")
    parser.add_argument("latentIndex", help="Latent index", type=int)
    parser.add_argument("trialIndex", help="Trial index", type=int)
    parser.add_argument("--latentMeanFilenamePattern", help="Latent mean filename pattern", default="results/simulations/{:08d}_latent{:d}_trial{:d}_latent_mean_{:s}.npz")
    parser.add_argument("--simInitConfigFilenamePattern", help="Simulation init filename pattern", default="data/{:08d}_simulation_metaData.ini")
    parser.add_argument("--variationalMeanFilenamePattern", help="Variational mean filename pattern", default="data/{:08d}_variational_mean_latent{:d}_trial{:d}.csv")
    args = parser.parse_args()

    simInitConfigNumber = args.simInitConfigNumber
    latentDescriptor = args.latentDescriptor
    k = args.latentIndex
    r = args.trialIndex
    latentMeanFilenamePattern = args.latentMeanFilenamePattern
    sim_init_config_filename_pattern = args.simInitConfigFilenamePattern
    variational_mean_filename_pattern = args.variationalMeanFilenamePattern

    variational_mean_filename = variational_mean_filename_pattern.format(simInitConfigNumber, k, r)
    latent_mean_filename = latentMeanFilenamePattern.format(simInitConfigNumber, k, r, latentDescriptor)
    loadRes = np.load(latent_mean_filename)
    latent_mean = torch.from_numpy(loadRes["latent_mean"])

    # load data and initial values
    sim_init_config_filename = sim_init_config_filename_pattern.format(simInitConfigNumber)
    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(sim_init_config_filename)

    trials_lengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    t0 = 0.0
    tf = trials_lengths[r]
    dt = float(simInitConfig["control_variables"]["dtCIF"])
    t = torch.arange(t0, tf, dt)

    nLatents = int(simInitConfig["control_variables"]["nLatents"])
    nTrials = len(trials_lengths)
    inducing_points_locs = utils.svGPFA.configUtils.getIndPointsLocs0(nLatents=nLatents, nTrials=nTrials, config=simInitConfig)
    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=False)

    tVarMean =  simulations.svGPFA.utils.getTrueVariationalMean(t=t,
                                                                latent_mean=latent_mean,
                                                                inducing_points_locs=inducing_points_locs[k][r,:,0],
                                                                kernel=kernels[k])
    np.savetxt(variational_mean_filename, tVarMean.numpy())

    Ktz = kernels[k].buildKernelMatrix(X1=t, X2=inducing_points_locs[k][r,:,0])
    Kzz = kernels[k].buildKernelMatrix(X1=inducing_points_locs[k][r,:,0],
                                       X2=inducing_points_locs[k][r,:,0])
    v = torch.linalg.solve(Kzz, tVarMean)
    latent_mean_approx = Ktz.matmul(v)

    print("Kzz condition number: {:f}".format(np.linalg.cond(Kzz)))

    plt.plot(t, latent_mean, label="true")
    plt.plot(t, latent_mean_approx, label="approximation")
    plt.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

