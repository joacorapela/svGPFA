# Python implementation of Sparse Variational Gaussian Process Factor Analysis (svGPFA, [Duncker and Sahani, 2018](https://papers.nips.cc/paper/2018/file/d1ff1ec86b62cd5f3903ff19c3a326b2-Paper.pdf)) ![tests](https://github.com/joacorapela/svGPFA/actions/workflows/tests.yml/badge.svg?branch=master) ![docs](https://github.com/joacorapela/svGPFA/actions/workflows/documentation.yml/badge.svg?branch=master)

* Documentation can be found [here](https://joacorapela.github.io/svGPFA/)
* A Colab notebook can be found [here](https://colab.research.google.com/github/joacorapela/svGPFA/blob/master/doc/ipynb/doEstimateAndPlot.ipynb) and a Jupyter notebook [here](docs/ipynb/doEstimateAndPlot.ipynb)
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

