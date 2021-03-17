#!/bin/bash

ipython --pdb doPlotPeriodParamVsCovariate.py -- \
    --pythonDescriptorsFilename ../data/descriptors_20secSim.csv \
    --matlabDescriptorsFilename ../data/descriptors_20secSim.csv \
    --labelsAndEstNumbersFilename_PyTorch_LBFGS ../data/labelsEstNumbers_20secSim_PyTorch_LBFGS.csv \
    --labelsAndEstNumbersFilename_SciPy_L-BFGS-B ../data/labelsEstNumbers_20secSim_SciPy_L-BFGS-B.csv \
    --labelsAndEstNumbersFilename_SciPy_trust-ncg ../data/labelsEstNumbers_20secSim_SciPy_trust-ncg.csv \
    --labelsAndEstNumbersFilename_Matlab_minFunc ../../../matlabCode/slurm/data/labelsEstNumbers_20secSim.csv \
    --generativePeriod 5.0 \
    --pythonPeriodScale 1000 \
    --pythonModelFilenamePattern ../../scripts/{:s} \
    --matlabModelFilenamePattern ../../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat \
    --figFilenamePattern ../figures/20secSim_periodParams.{:s} \
    --xlab "Period Parameter Initial Condition" \
    --ylab "Estimated Period Parameter" \
