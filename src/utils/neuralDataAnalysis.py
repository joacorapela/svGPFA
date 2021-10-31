
import torch

def binSpikes(spikes, bin_edges):
    bin_counts, bin_edges_output = torch.histogram(input=spikes, bins=bin_edges)
    bin_centers = (bin_edges_output[:-1]+bin_edges_output[1:])/2.0
    return bin_counts, bin_centers
