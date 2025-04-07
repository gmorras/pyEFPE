# pyEFPE

`pyEFPE` is a Python-based post-Newtonian waveform model for inspiralling precessing-eccentric compact binaries.

## References

If you use `pyEFPE` in your research, please cite:

G. Morras, G. Pratten, and P. Schmidt, _pyEFPE: An improved post-Newtonian waveform model for inspiralling precessing-eccentric compact binaries_, 2025, [arXiv:2502.03929](https://arxiv.org/abs/2502.03929).

You can use the following BibTeX entry:

```bibtex
@article{Morras:2025nlp,
    author = "Morras, Gonzalo and Pratten, Geraint and Schmidt, Patricia",
    title = "{pyEFPE: An improved post-Newtonian waveform model for inspiralling precessing-eccentric compact binaries}",
    eprint = "2502.03929",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    reportNumber = "IFT-UAM/CSIC-25-12",
    month = "2",
    year = "2025"
}
```

## Installation

To install `pyEFPE`, navigate to the root directory of the repository (where `setup.py` is located) and run:

```bash
pip install .
```

## Getting Started

To generate a frequency-domain waveform using pyEFPE, try the following example:

```python
# Import required packages
import pyEFPE
import numpy as np

# Define binary parameters (for additional details see pyEFPE/waveform/EFPE.py)
params = {
    'mass1': 2.4,       # Mass of companion 1 (solar masses)
    'mass2': 1.2,       # Mass of companion 2 (solar masses)
    'e_start': 0.7,     # Initial eccentricity
    'spin1x': -0.44,    # Spin components of companion 1
    'spin1y': -0.26,
    'spin1z': 0.48,
    'spin2x': -0.31,    # Spin components of companion 2
    'spin2y': 0.01,
    'spin2z': -0.84,
    'inclination': 1.57,# Initial binary inclination (radians)
    'f22_start': 10,    # Starting (simulation) waveform frequency of GW 22 mode (Hz)
}

# Initialize pyEFPE waveform model
wf = pyEFPE.pyEFPE(params)

# Define frequency array for waveform generation
freqs = np.arange(20, 1024, 1/128)

# Compute frequency-domain gravitational wave polarizations
hp, hc = wf.generate_waveform(freqs)
```

You can visualize this waveform using the following code:

```python
from matplotlib import pyplot as plt

plt.figure()
plt.loglog(freqs,  np.abs(hp), 'C0-' , label=r'$|\tilde{h}_+|$')
plt.loglog(freqs,  np.abs(hc), 'C1-' , label=r'$|\tilde{h}_\times|$')
plt.xlabel(r'$f$ [Hz]')
plt.ylabel(r'Frequency domain strain $[\mathrm{Hz}^{-1}]$')
plt.legend()
plt.show()
```

## Repository Overview

The installable waveform model is located in the [pyEFPE/](pyEFPE) folder, while the [Tests/](Tests) folder contains various examples and tests of the waveform. Most notably, the [Tests/model_validation/](Tests/model_validation) directory contains the scripts used to generate most of the model-validation figures from [the pyEFPE paper](https://arxiv.org/abs/2502.03929), and the [Tests/test_theory/](Tests/test_theory) directory contains Mathematica notebooks and Python scripts used to validate some of the post-Newtonian expressions that the model is based on.

## Feedback & Issues

We welcome feedback, bug reports, and feature requests! If you encounter any issues or have suggestions for improvements, please open an issue in the GitHub issue tracker.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

