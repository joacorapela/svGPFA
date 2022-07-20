
import sys
import pdb
import argparse
import configparser
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("simInitConfigNumber", help="Simulation initialization configuration number", type=int)
    parser.add_argument("latentIndex", help="Latent index", type=int, default=0)
    parser.add_argument("trialIndex", help="Trial index", type=int, default=0)
    parser.add_argument("--simInitConfigFilenamePattern", help="Simulation init filename pattern", default="data/{:08d}_simulation_metaData.ini")
    parser.add_argument("--latentMeanFilenamePattern", help="Latent mean filename pattern", default="results/simulations/{:08d}_latent{:d}_trial{:d}_latent_mean_sigmoidal.npz")
    parser.add_argument("--latentMeanFigFilenamePattern", help="Latent mean figure filename pattern", default="figures/simulations/{:08d}_latent{:d}_trial{:d}_latent_mean_sigmoidal.png")
    args = parser.parse_args()

    simInitConfigNumber = args.simInitConfigNumber
    k = args.latentIndex
    r = args.trialIndex
    sim_init_config_filename_pattern = args.simInitConfigFilenamePattern
    latent_mean_filename_pattern = args.latentMeanFilenamePattern
    latent_mean_figure_filename_pattern = args.latentMeanFigFilenamePattern

    sim_init_config_filename = sim_init_config_filename_pattern.format(simInitConfigNumber)
    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(sim_init_config_filename)

    trials_lengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    t0 = 0.0
    tf = trials_lengths[r]
    dt = float(simInitConfig["control_variables"]["dtCIF"])

    t = np.arange(t0, tf, dt)
    latent_mean = 1.0/(1+np.exp(-((t-t0)*12.0/(tf-t0)-6.0)))

    latent_mean_filename = latent_mean_filename_pattern.format(simInitConfigNumber, k, r)
    np.savez(latent_mean_filename, latent_mean=latent_mean)

    latent_mean_figure_filename = latent_mean_figure_filename_pattern.format(simInitConfigNumber, k, r)
    plt.plot(t, latent_mean)
    plt.xlabel("Time (sec)")
    plt.ylabel("Latent Mean Value")
    plt.savefig(latent_mean_figure_filename)
    plt.show()

    pdb.set_trace()
if __name__=="__main__":
    main(sys.argv)
