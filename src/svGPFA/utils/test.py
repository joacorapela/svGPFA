
import torch

def getDefaultParamsDict(n_neurons, n_trials, n_latents=3,
                         n_ind_points=None, common_n_ind_points=10, n_quad=200,
                         trials_start_time=0.0, trials_end_time=1.0,
                         diag_var_cov0_value=1e-2, prior_cov_reg_param=1e-3,
                         lengthscale=1.0, em_max_iter=50):
    if n_ind_points is None:
        n_ind_points = [common_n_ind_points] * n_latents
    var_mean0 = [torch.zeros((n_trials, n_ind_points[k], 1),
                             dtype=torch.double)
                 for k in range(n_latents)]
    var_cov0 = [[] for r in range(n_latents)]
    for k in range(n_latents):
        var_cov0[k] = torch.empty((n_trials, n_ind_points[k], n_ind_points[k]),
                                  dtype=torch.double)
        for r in range(n_trials):
            var_cov0[k][r, :, :] = torch.eye(n_ind_points[k]) * \
                diag_var_cov0_value

    params_dict = {
        "data_structure_params": {"trials_start_time": trials_start_time,
                                  "trials_end_time": trials_end_time},
        "variational_params0": {
            "variational_mean0": var_mean0,
            "variational_cov0": var_cov0,
        },
        "embedding_params0": {
            "c0_distribution": "Normal",
            "c0_loc": 0.0,
            "c0_scale": 1.0,
            "d0_distribution": "Normal",
            "d0_loc": 0.0,
            "d0_scale": 1.0,
        },
        "kernels_params0": {
            "k_types": "exponentialQuadratic",
            "k_lengthscales0": lengthscale,
        },
        "ind_points_locs_params0": {
            "n_ind_points": n_ind_points,
            "ind_points_locs0_layout": "equidistant",
        },
        "optim_params": {
            "n_quad": n_quad,
            "prior_cov_reg_param": prior_cov_reg_param,
            "optim_method": "ecm",
            "em_max_iter": em_max_iter,
            "verbose": True,
            #
            "estep_estimate": True,
            "estep_max_iter": 20,
            "estep_lr": 1.0,
            "estep_tolerance_grad": 1e-7,
            "estep_tolerance_change": 1e-9,
            "estep_line_search_fn": "strong_wolfe",
            #
            "mstep_embedding_estimate": True,
            "mstep_embedding_max_iter": 20,
            "mstep_embedding_lr": 1.0,
            "mstep_embedding_tolerance_grad": 1e-7,
            "mstep_embedding_tolerance_change": 1e-9,
            "mstep_embedding_line_search_fn": "strong_wolfe",
            #
            "mstep_kernels_estimate": True,
            "mstep_kernels_max_iter": 20,
            "mstep_kernels_lr": 1.0,
            "mstep_kernels_tolerance_grad": 1e-7,
            "mstep_kernels_tolerance_change": 1e-9,
            "mstep_kernels_line_search_fn": "strong_wolfe",
            #
            "mstep_indpointslocs_estimate": True,
            "mstep_indpointslocs_max_iter": 20,
            "mstep_indpointslocs_lr": 1.0,
            "mstep_indpointslocs_tolerance_grad": 1e-7,
            "mstep_indpointslocs_tolerance_change": 1e-9,
            "mstep_indpointslocs_line_search_fn": "strong_wolfe"}
    }
    return params_dict
