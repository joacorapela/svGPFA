import sys
import pdb
import torch
import pickle
import argparse
import configparser

sys.path.append("../src")
import utils.neuralDataAnalysis

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_res_number", help="simuluation result number", type=int)
    parser.add_argument("bin_width", help="bin width (sec)", type=float)
    parser.add_argument("--sim_config_filename_pattern", 
                        help="simulation configuration filename pattern",
                        default="results/{:08d}_simulation_metaData.ini")
    parser.add_argument("--results_filename_pattern", 
                        help="results filename_pattern",
                        default="results/{:08d}_binned_spikes_binWidth{:.02f}.pickle")
    args = parser.parse_args()

    sim_res_number = args.sim_res_number
    bin_width = args.bin_width
    sim_config_filename_pattern = args.sim_config_filename_pattern
    results_filename_pattern = args.results_filename_pattern

    sim_config_filename = sim_config_filename_pattern.format(sim_res_number)
    results_filename = results_filename_pattern.format(sim_res_number,
                                                       bin_width)

    sim_res_config = configparser.ConfigParser()
    sim_res_config.read(sim_config_filename)
    sim_init_config_filename = sim_res_config["simulation_params"]["simInitConfigFilename"]
    sim_res_filename = sim_res_config["simulation_results"]["simResFilename"]

    sim_init_config = configparser.ConfigParser()
    sim_init_config.read(sim_init_config_filename)
    trials_lengths = [float(str) for str in sim_init_config["control_variables"]["trialsLengths"][1:-1].split(",")]

    with open(sim_res_filename, "rb") as f: simRes = pickle.load(f)
    spikes_times = simRes["spikes"]
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    # we need to fix the following line and to use trials of variable length
    bin_edges = torch.arange(0.0, trials_lengths[0]+bin_width/2.0, bin_width)
    n_bins = len(bin_edges)-1
    bin_counts = torch.empty((n_trials, n_bins, n_neurons), dtype=torch.int)

    for r in range(len(spikes_times)):
        for n in range(len(spikes_times[r])):
            bin_counts[r,:,n], bin_times = \
                utils.neuralDataAnalysis.binSpikes(spikes=spikes_times[r][n],
                                                   bin_edges=bin_edges)

    results_to_save = {"bin_counts": bin_counts, "bin_times": bin_times}

    with open(results_filename, "wb") as f: pickle.dump(results_to_save, f)
    print("Saved results to {:s}".format(results_filename))

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
