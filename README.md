# SREPSim

Cite as follows:
``` bibtex
@INPROCEEDINGS{srepsim_icbc2023,
  author={Boškov, Novak and and Simsek, Sevval and Trachtenberg, Ari and Starobinski, David},
  booktitle={2023 IEEE International Conference on Blockchain and Cryptocurrency (ICBC)},
  title={SREP: Out-Of-Band Sync of Transaction Pools for Large-Scale Blockchains},
  year={2023},
  volume={},
  number={},
  pages={},
  doi={}
}
```

## Instructions

Use a virtualenv and Python 3.9.

Install [Nauty](http://users.cecs.anu.edu.au/~bdm/nauty/) to generate
non-isomorphic graphs given the number of vertices.

``` shell
$ pip install -e .
```

Optionally:

``` shell
$ pytest
```

Use as follows:

``` python
import srep_simulator
srep_simulator.aux.DEFAULT_NAUTY_PATH = 'PATH_TO_YOUR_INSTALL_DIR'

sim = srep_simulator.SREPSimulator()
sim.run()
```

See the in-source documentation.
