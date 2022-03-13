import pdb
import os
import numpy as np
import scipy.io
import torch

def  getTrialAndLocationSpikesTimes(mat, trial_index, location):
    mat_spikes_times = mat["Rb"][trial_index, location]['unit']['spikeTimes']
    spikes_times = []
    nNeurons = mat_spikes_times.shape[1]
    for j in range(nNeurons):
        spikes_times_set = (mat_spikes_times[0,j].astype(np.float64)).squeeze()
        spikes_times.append(torch.from_numpy(spikes_times_set))
    return spikes_times

def getTrialsAndLocationSpikesTimes(mat, trials_indices, location):
    spikesTimes = [[]]*len(trials_indices)
    for i, trial_index in enumerate(trials_indices):
        trial_spikes_times = getTrialAndLocationSpikesTimes(mat=mat,
                                                            trial_index=
                                                             trial_index, 
                                                            location=location)
        spikesTimes[i] = trial_spikes_times
    return spikesTimes

def getSpikesTimes(data_filename, trials_indices, location, from_time, to_time, min_nSpikes_perNeuron_perTrial):
    mat = scipy.io.loadmat(os.path.expanduser(data_filename))
    spikes_times = getTrialsAndLocationSpikesTimes(mat=mat,
                                                   trials_indices=trials_indices,
                                                   location=location)
    spikes_times = clipSpikesTimes(spikes_times=spikes_times,
                                               from_time=from_time,
                                               to_time=to_time)
    return spikes_times

