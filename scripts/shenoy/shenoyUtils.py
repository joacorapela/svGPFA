import numpy as np
import torch

def  getTrialAndLocationSpikesTimes(mat, trial, location, sRate=1000):
    mat_spike_times = mat["Rb"][trial, location]['unit']['spikeTimes']
    spikes_times = []
    for j in range(mat_spike_times.shape[1]):
        spikes_times_set = (mat_spike_times[0,j].astype(np.float64)/sRate).squeeze()
        spikes_times.append(torch.from_numpy(spikes_times_set))
    return spikes_times

def getTrialsAndLocationSpikesTimes(mat, trials, location, sRate=1000):
    spikesTimes = [[]]*len(trials)
    for i, trial in enumerate(trials):
        trial_spikes_times = getTrialAndLocationSpikesTimes(mat=mat, trial=trial, 
                                                            location=location,
                                                            sRate=sRate)
        spikesTimes[i] = trial_spikes_times
    return spikesTimes
