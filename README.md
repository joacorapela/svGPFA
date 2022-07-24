# Python implementation of Sparse Variational Gaussian Process Factor Analysis (svGPFA, [Duncker and Sahani, 2018](https://papers.nips.cc/paper/2018/file/d1ff1ec86b62cd5f3903ff19c3a326b2-Paper.pdf)) ![tests](https://github.com/joacorapela/svGPFA/actions/workflows/tests.yml/badge.svg?branch=master) ![docs](https://readthedocs.org/projects/pip/badge/)

* Documentation can be found [here](https://svgpfa.readthedocs.io/)
* A script running svGPFA on simulated data can be found here [here](scripts/demoPointProcessLeasSimulation-noGPU.py)
* A Colab notebook can be found [here](https://colab.research.google.com/drive/1Ze60RlX65-Yx8oG1EdKYm2mSvVCMaJgv?usp=sharing) and a Jupyter notebook [here](docs/ipynb/doEstimateAndPlot.ipynb)
<!---
* A Dash/Plotly GUI can be found [here](gui/doRunGUI.py)
* The source code can be found under [src](src)
* Test cases can be found under [ci](ci) and the history of running these test can be found [here](https://github.com/joacorapela/svGPFA/actions).
--->

# Installation

1. clone this repo

2. if you will *not* test the svGPFA installation with the example below (see section *Verify installation* below) and will not run the example notebooks (see above), in the root directory of the cloned repo type

    ```
    pip install -e .
    ```
    But if you will test the application with the example script provided below, or will run the example notebooks (see above), in the root directory of the cloned repo type

     ```
     pip install -e .[examples]
     ```

# Testing the installation

1. From the root directory of the cloned svGPFA directory, change the current directory to *examples/scripts*.

    ```
    cd examples/scripts
    ```

2. run the estimation of svGPFA parameters (for only two EM iterations)

    ```
    python doEstimateSVGPFA --max_iter=2
    ```

3. if everything went well you should see a newly created estimation result file *../results/xxxxxxxx_etimationRes.pickle*.

