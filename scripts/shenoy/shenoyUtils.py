import pdb
import numpy as np
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

def clipSpikesTimes(spikes_times, from_time, to_time):
    nTrials = len(spikes_times)
    clipped_spikes_times = [[]] * nTrials
    for r in range(nTrials):
        clipped_spikes_times[r] = clipTrialSpikesTimes(trial_spikes_times=spikes_times[r],
                                                       from_time=from_time,
                                                       to_time=to_time)
    return clipped_spikes_times

def clipTrialSpikesTimes(trial_spikes_times, from_time, to_time):
    nNeurons = len(trial_spikes_times)
    clipped_trial_spikes_times = [[]] * nNeurons
    for n in range(nNeurons):
        clipped_trial_spikes_times[n] = clipNeuronSpikesTimes(
            neuron_spikes_times=trial_spikes_times[n],
            from_time=from_time, to_time=to_time)
    return clipped_trial_spikes_times

def clipNeuronSpikesTimes(neuron_spikes_times, from_time, to_time):
    clipped_neuron_spikes_times = neuron_spikes_times[torch.logical_and(
        from_time <= neuron_spikes_times, neuron_spikes_times < to_time)]
    return clipped_neuron_spikes_times

def offsetSpikeTimes(spikes_times, offset):
    nTrials = len(spikes_times)
    offsetted_spikes_times = [[]] * nTrials
    for r in range(nTrials):
        offsetted_spikes_times[r] = offsetTrialSpikesTimes(trial_spikes_times=spikes_times[r], offset=offset)
    return offsetted_spikes_times

def offsetTrialSpikesTimes(trial_spikes_times, offset):
    nNeurons = len(trial_spikes_times)
    offsetted_trial_spikes_times = [[]] * nNeurons
    for n in range(nNeurons):
        offsetted_trial_spikes_times[n] = trial_spikes_times[n]+offset
    return offsetted_trial_spikes_times

def removeUnits(spikes_times, units_to_remove):
    nTrials = len(spikes_times)
    spikes_times_woUnits = [[]] * nTrials
    for r in range(nTrials):
        spikes_times_woUnits[r] = \
                removeUnitsFromTrial(trial_spikes_times=spikes_times[r],
                                     units_to_remove=units_to_remove)
    return spikes_times_woUnits

def removeUnitsFromTrial(trial_spikes_times, units_to_remove):
    nNeurons = len(trial_spikes_times)
    spikes_times_woUnits = []
    for n in range(nNeurons):
        if n not in units_to_remove:
            spikes_times_woUnits.append(trial_spikes_times[n])
    return spikes_times_woUnits

def selectUnitsWithLessSpikesThanThrInAllTrials(spikes_times, thr):
    nTrials = len(spikes_times)
    nNeurons = len(spikes_times[0])
    selected_units = set([i for i in range(nNeurons)])
    for r in range(nTrials):
        selected_trial_units = selectUnitsWithLessSpikesThanThrInTrial(
            spikes_times=spikes_times[r], thr=thr)
        selected_units = selected_units.intersection(selected_trial_units)
    answer = list(selected_units)
    return answer

def selectUnitsWithLessSpikesThanThrInAnyTrial(spikes_times, thr):
    nTrials = len(spikes_times)
    selected_units = set([])
    for r in range(nTrials):
        selected_trial_units = selectUnitsWithLessSpikesThanThrInTrial(
            spikes_times=spikes_times[r], thr=thr)
        selected_units = selected_units.union(selected_trial_units)
    answer = list(selected_units)
    return answer
    return selected_units

def selectUnitsWithLessSpikesThanThrInTrial(spikes_times, thr):
    nNeurons = len(spikes_times)
    selected_units = set([])
    for n in range(nNeurons):
        if len(spikes_times[n])<thr:
            selected_units.add(n)
    return selected_units

