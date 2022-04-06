# Python implementation of Sparse Variational Gaussian Process Factor Analysis (svGPFA, [Duncker and Sahani, 2018](https://papers.nips.cc/paper/2018/file/d1ff1ec86b62cd5f3903ff19c3a326b2-Paper.pdf)) ![tests](https://github.com/joacorapela/svGPFA/actions/workflows/tests.yml/badge.svg?branch=master) ![docs](https://readthedocs.org/projects/pip/badge/)

Tested with [Python 3](https://www.python.org/downloads/release/python-352/) and [Pytorch 1.3.0](https://pytorch.org/).

* A Colab notebook can be found [here](https://colab.research.google.com/drive/1iOMZYBu4DMFYayXlgrm_nFAKge3xIkKr?usp=sharing) and a Jupyter notebook [here](ipynb/demoPointProcess.ipynb)
* Documentation can be found [here](https://svgpfa.readthedocs.io/)
* A Dash/Plotly GUI can be found [here](gui/doRunGUI.py)
* A script running svGPFA on simulated data can be found here [here](scripts/demoPointProcessLeasSimulation-noGPU.py)
* The source code can be found under [src](src)
* Test cases can be found under [ci](ci) and the history of running these test can be accessed by clicking on the badge in the title
