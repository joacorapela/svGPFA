import numpy as np

def bootstrapMean(observations, nResamples):
    indices = np.arange(len(observations))
    bootstrapped_means = np.empty(nResamples)
    for i in range(nResamples):
        sample_indices = np.random.choice(indices, size=len(indices), replace=True)
        resampled_observations = observations[sample_indices]
        if remove_nan:
            resampled_observations = \
                resampled_observations[~np.isnan(resampled_observations)]
        bootstrapped_means[i] = np.means(resampled_observations)
    return bootstrapped_means

def bootstrapMedian(observations, nResamples):
    indices = np.arange(len(observations))
    bootstrapped_medians = np.empty(nResamples)
    for i in range(nResamples):
        sample_indices = np.random.choice(indices, size=len(indices), replace=True)
        resampled_observations = observations[sample_indices]
        bootstrapped_medians[i] = np.median(resampled_observations)
    return bootstrapped_medians

def estimatePercentileCI(alpha, bootstrapped_stats):
    sortedSamples = np.sort(bootstrapped_stats)
    N = len(bootstrapped_stats)
    ciLow = sortedSamples[int(alpha/2*N)]
    ciHigh = sortedSamples[int((1-alpha/2)*N)]
    ci = (ciLow, ciHigh)
#     if ciLow>0:
#         import pdb; pdb.set_trace()
    return ci
