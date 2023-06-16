# PhaseUtils
*A selection of utilities to retrieve and process phase information of optical fields*

## Dependencies

- `numba`
- `cupy`
- `numpy`
- `skimage`
- `numbalsoda`
- `pyfftw`

## `contrast.py`

This code contains all the functions necessary to retrieve the phase from an off-axis interferogram as presented in [this preprint](https://arxiv.org/abs/2202.05764).
It leverages the computation capabilities of GPU's to achieve ms speed for phase recovery, which allows it to be used in real time.
It also includes fucntions (`delta_n`) to recover the non-linear index of some medium by measuring phase front deformation through Kerr effect.
Most of the functions can run on the CPU or the GPU: the GPU variant of each function is indicated by its `_cp` postfix.
