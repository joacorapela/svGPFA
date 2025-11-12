import jax.numpy as jnp
import svGPFA.utils.miscUtils
import svGPFA.stats.kernelsMatricesStore
import svGPFA.stats.posteriorOnLatents
import svGPFA.stats.preIntensity

def computeLatents(vMean, vChol, kernels_params, ind_points_locs,
                   kernels_types, leg_quad_points, reg_param):
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params)
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol
    indPointsLocsKMS.init(kernels=kernels)
    quadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndQuadTimesKMS
    quadTimesKMS.init(kernels=kernels, t=leg_quad_points)
    Kzz, Kzz_cho = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params, ind_points_locs=ind_points_locs,
        reg_param=reg_param)
    Ktz_quad = quadTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params, ind_points_locs=ind_points_locs)
    l_means = svGPFA.stats.posteriorOnLatents.PosteriorOnLatents.computeMeans(
        vMean=vMean, Kzz_cho=Kzz_cho, Ktz=Ktz_quad)
    vCov = svGPFA.utils.miscUtils.buildCovsFromCholVecs(vChol)
    l_vars = svGPFA.stats.posteriorOnLatents.PosteriorOnLatents.computeVars(
        vCov=vCov, Kzz=Kzz, Kzz_cho=Kzz_cho, Ktz=Ktz_quad)
    return l_means, l_vars


def computePreIntensity(C, d, vMean, vChol, kernels_params, ind_points_locs,
                        kernels_types, leg_quad_points, reg_param):
    # h_means, h_vars \in n_trials x n_neurons x n_quad
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params)
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol
    indPointsLocsKMS.init(kernels=kernels)
    quadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndQuadTimesKMS
    quadTimesKMS.init(kernels=kernels, t=leg_quad_points)
    Kzz, Kzz_cho = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params, ind_points_locs=ind_points_locs,
        reg_param=reg_param)
    Ktz_quad = quadTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params, ind_points_locs=ind_points_locs)
    h_means = svGPFA.stats.preIntensity.LinearPreIntensity.computeMeans(
        vMean=vMean, C=C, d=d, Kzz_cho=Kzz_cho, Ktz=Ktz_quad)
    vCov = svGPFA.utils.miscUtils.buildCovsFromCholVecs(vChol)
    h_vars = svGPFA.stats.preIntensity.LinearPreIntensity.computeVars(
        vCov=vCov, C=C, Kzz=Kzz, Kzz_cho=Kzz_cho, Ktz=Ktz_quad)
    return h_means, h_vars

def computeCIFmeanAndSTD(h_means, h_vars):
    # h_means, h_vars \in n_trials x n_neurons x n_quad
    # answer \in n_trials x n_neurons x n_quad
    mean = jnp.exp(h_means + 0.5 * h_vars)
    std = mean*jnp.sqrt(jnp.exp(h_vars)-1)
    return mean, std
