# Comprehensive Library of Variational LSE Solvers

This repo contains the code for the `qiskit-torch-module` introduced in 
"Qiskit-Torch-Module: Fast Prototyping of Quantum Neural Networks", N. Meyer et al. (2024).

## Setup and Installation

The library requires an installation of `python 3.12`, and following libraries:
- `qiskit~=1.0.0`, backward compatible up to `qiskit v0.44.0`
- `qiskit-algorithms~=0.3.0`
- `torch~=2.2.1`
- `threadpoolctl~=3.3.0`

We recommend setting up a conda environment:

```
conda create --name ENV_NAME python=3.12
conda activate ENV_NAME
```

The package `qiskit-torch-module` can be installed locally via:
```
cd qiskit-torch-module
pip install -e .
```

## Usage and Reproduction of Results

Information on how to use the different modalities of the libraries are described in the documentation.
Additionally, we provide to usage examples.

To run the benchmarks and end-to-end implementations, the additional libraries in `examples/requirements.txt` need to be installed:
- `gymnasium~=0.29.1`
- `qiskit-machine-learning~=0.7.1`
- `tqdm~=4.66.2`
- `torchvision~=0.17.1`

This can also be done with
```
pip install -r examples/requirements.txt
```

### Benchmarking of Raw Runtimes

This allows for benchmarking of runtimes for forward and backward pass compared to `qiskit-machine-learning`:

```
python examples/benchmark.py --qubits 12 --depth 3 --thread 0 [--use_qml]
```

This benchmarks the performance for 12 qubits with a circuit depth of 3. 
Appending / removing the ``--use_qml`` flag defines the usage of ``qiskit-machine-learning`` / ``qiskit-torch-module``.
It is possible to select an explicit number of parallel workers, with ``threads=0`` using all available CPU cores.

### End-to-End Quantum Machine Learning Pipelines

#### Classification

This implements the full-quantum classification algorithm proposed in
["Incremental Data-Uploading for Full-Quantum Classification", M. Periyasamy et al., IEEE QCE 1:31-37 (2022)](https://ieeexplore.ieee.org/document/9951318)

```
python examples/qml.py [--use_qml]
```


#### Quantum Reinforcement Learning

This implements the quantum policy gradient algorithm proposed in
["Quantum Policy Gradient Algorithm with Optimized Action Decoding", N. Meyer et al., PMLR 202:24592-24613 (2023)](https://proceedings.mlr.press/v202/meyer23a.html):

```
python examples/qrl.py [--use_qml]
```

## Acknowledgements

The backbone of our implementation is the ``qiskit`` software framework: https://github.com/Qiskit

Furthermore, we git inspired by ``qiskit-machine-learning``: https://github.com/qiskit-community/qiskit-machine-learning

## Citation

If you use the `qiskit-torch-module` or results from the paper, please cite our work as

```
@article{meyer2024qiskit,
  title={Qiskit-Torch-Module: Fast Prototyping of Quantum Neural Networks},
  author={Meyer, Nico and Ufrecht, Christian and Periyasamy, Maniraman and Plinge, Axel and Mutschler, Christopher and Scherer, Daniel D. and Maier, Andreas},
  journal={arXiv:...},
  year={2024},
  doi={.../arXiv...}
}
```

## Version History

Initial release (v1.0): April 2024

## License

Apache 2.0 License
  