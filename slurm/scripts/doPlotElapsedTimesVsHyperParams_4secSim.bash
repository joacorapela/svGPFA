#!/bin/bash

# ipython --pdb doPlotElapsedTimeVsCovariate.py -- \
python doPlotElapsedTimeVsCovariate.py \
    --pythonDescriptorsFilename ../data/descriptors_4secSim.csv \
    --matlabDescriptorsFilename ../../../matlabCode/slurm/data/descriptors_4secSim.csv \
    --labelsAndEstNumbersFilename_PyTorch_LBFGS ../data/labelsEstNumbers_4secSim_PyTorch_LBFGS.csv \
    --labelsAndEstNumbersFilename_SciPy_L-BFGS-B ../data/labelsEstNumbers_4secSim_SciPy_L-BFGS-B.csv \
    --labelsAndEstNumbersFilename_SciPy_trust-ncg ../data/labelsEstNumbers_4secSim_SciPy_trust-ncg.csv \
    --labelsAndEstNumbersFilename_Matlab_minFunc ../../../matlabCode/slurm/data/labelsEstNumbers_4secSim.csv \
    --pythonModelFilenamePattern ../../scripts/{:s} \
    --matlabModelFilenamePattern ../../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat \
    --convergenceTolerance 1e-5 \
    --figFilenamePattern ../figures/4secSim_elapsedTimes.{:s} \
    --xlab "Hyper Parameter Value" \
    --ylab "Elapsed Time (sec)"
