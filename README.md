# Python implementation of Sparse Variational Gaussian Process Factor Analysis (svGPFA, [Duncker and Sahani, 2018](https://papers.nips.cc/paper/2018/file/d1ff1ec86b62cd5f3903ff19c3a326b2-Paper.pdf)) ![tests](https://github.com/joacorapela/svGPFA/actions/workflows/tests.yml/badge.svg?branch=master) ![docs](https://github.com/joacorapela/svGPFA/actions/workflows/documentation.yml/badge.svg?branch=master)

svGPFA identifies common latent structure in neural population spike-trains.
It uses shared latent Gaussian processes, which are combined linearly as in
Gaussian Process Factor Analysis (GPFA, [Yu et al., 2009](https://journals.physiology.org/doi/full/10.1152/jn.90941.2008?rfr_dat=cr_pub++0pubmed&url_ver=Z39.88-2003&rfr_id=ori%3Arid%3Acrossref.org)).
svGPFA extends GPFA to handle unbinned spike-train data by using a continuous
time point-process likelihood model and achieving scalability using a sparse
variational approximation.

You can run svGPFA on sample data, plot its estimates and perform goodness-of-fit tests (without installing anything in your computer) by just running [this](https://colab.research.google.com/drive/1Ze60RlX65-Yx8oG1EdKYm2mSvVCMaJgv) Google Colab notebook.
You can also do this by installing svGPFA (intructions [below](#installation)) and running [this](doc/ipynb/doEstimateAndPlot.ipynb) Jupyter notebok.
In addition, after installing svGPFA, you can estimate models using a script, as shown [below](#testing-the-installation).

Documentation can be found [here](https://joacorapela.github.io/svGPFA/).

<!---
* A script running svGPFA on simulated data can be found here [here](scripts/demoPointProcessLeasSimulation-noGPU.py)
* A Dash/Plotly GUI can be found [here](gui/doRunGUI.py)
* The source code can be found under [src](src)
* Test cases can be found under [ci](ci) and the history of running these test can be found [here](https://github.com/joacorapela/svGPFA/actions).
--->

# Installation

1. clone this repo

2. change the current directory to that of the cloned repo

    ```
    cd svGPFA
    ```

3. if you will *not* run the example notebooks (see above), in the root directory of the cloned repo type

    ```
    pip install -e .
    ```
    If you will run the example notebooks (see above), in the root directory of the cloned repo type

     ```
     pip install -e .[notebook]
     ```

# Testing the installation

1. From the root directory of the cloned svGPFA directory, change the current directory to *examples/scripts*.

    ```
    cd examples/scripts
    ```

2. run the estimation of svGPFA parameters (for only two EM iterations)

    ```
    python doEstimateSVGPFA.py --em_max_iter=2
    ```

3. if everything went well the previous script should terminate after showing the following line in the standard output:

    ```
    Saved results to results/xxxxxxxx_etimationRes.pickle*.
    ```

