
import sys
import numpy as np
import torch

sys.path.append("../src")
import stats.pointProcess.sampling
import utils.neuralDataAnalysis

def test_thinning(plot=False):
    freq = 1.0 # Hz
    phase = torch.pi/2
    amplitude = 5.0
    nTrials = 100
    bin_size = 0.01
    tol = 100.0
    start_time = 0.0
    end_time = 1.0
    CIF_time_step = 1e-3
    psth_bin_size = 1e-2
    epoch_times = [0.0 for r in range(nTrials)]                                  

    CIF_times = torch.arange(start_time, end_time, CIF_time_step)
    psth_bins_edges = np.arange(start_time,
                               end_time+psth_bin_size,
                               psth_bin_size)
    CIF_values = torch.exp(amplitude*torch.cos(2*torch.pi*freq*CIF_times+phase))
    spikes_times = []
    for r in range(nTrials):
        spikes_times_one_neuron = \
            stats.pointProcess.sampling.sampleInhomogeneousPP_thinning(
                CIF_times=CIF_times, CIF_values=CIF_values)["inhomogeneous"]
        spikes_times.append([np.array(spikes_times_one_neuron)])
    binned_spikes, psth = utils.neuralDataAnalysis.computeBinnedSpikesAndPSTH(
            spikes_times=spikes_times, neuron_index=0,
            trials_indices=np.arange(nTrials),
            bins_edges=psth_bins_edges, time_unit="sec")
    psth_bin_centers = (psth_bins_edges[1:]+psth_bins_edges[:-1])/2
    CIF_values_interp = np.interp(x=psth_bin_centers, xp=CIF_times,
                                  fp=CIF_values)
    mse = np.mean((psth-CIF_values_interp)**2)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(psth_bin_centers, psth, label="PSTH")
        plt.plot(psth_bin_centers, CIF_values_interp, label="CIF")
        plt.legend()
        plt.title("MSE={:.02f}".format(mse))
        plt.xlabel("Time (sec)")
        plt.ylabel("Spike Rate (Hz)")
        import pdb; pdb.set_trace()
    assert(mse<tol)

def test_timeRescaling(plot=False):
    freq = 1.0 # Hz
    phase = torch.pi/2
    amplitude = 5.0
    nTrials = 100
    bin_size = 0.01
    tol = 70.0
    start_time = 0.0
    end_time = 1.0
    CIF_time_step = 1e-3
    psth_bin_size = 1e-2
    epoch_times = [0.0 for r in range(nTrials)]

    CIF_times = torch.arange(start_time, end_time, CIF_time_step)
    psth_bins_edges = np.arange(start_time,
                               end_time+psth_bin_size,
                               psth_bin_size)
    CIF_values = torch.exp(amplitude*torch.cos(2*torch.pi*freq*CIF_times+phase))
    spikes_times = []
    for r in range(nTrials):
        spikes_times_one_neuron = \
            stats.pointProcess.sampling.sampleInhomogeneousPP_timeRescaling(
                CIF_times=CIF_times, CIF_values=CIF_values)
        spikes_times.append([np.array(spikes_times_one_neuron)])
    binned_spikes, psth = utils.neuralDataAnalysis.computeBinnedSpikesAndPSTH(
            spikes_times=spikes_times, neuron_index=0,
            trials_indices=np.arange(nTrials),
            bins_edges=psth_bins_edges, time_unit="sec")
    psth_bin_centers = (psth_bins_edges[1:]+psth_bins_edges[:-1])/2
    CIF_values_interp = np.interp(x=psth_bin_centers, xp=CIF_times,
                                  fp=CIF_values)
    mse = np.mean((psth-CIF_values_interp)**2)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(psth_bin_centers, psth, label="PSTH")
        plt.plot(psth_bin_centers, CIF_values_interp, label="CIF")
        plt.legend()
        plt.title("MSE={:.02f}".format(mse))
        plt.xlabel("Time (sec)")
        plt.ylabel("Spike Rate (Hz)")
        import pdb; pdb.set_trace()
    assert(mse<tol)

if __name__ == "__main__":
    test_thinning(plot=True)
    test_timeRescaling(plot=True)
