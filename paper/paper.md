---
title: 'PhaseUtils: A Python package to process and control optical fields'
tags:
  - Python
authors:
  - name: Tangui Aladjidi
    orcid: 0000-0002-3109-9723
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Myrann Baker-Rasooli
    orcid: 0000-0003-0969-6705
    affiliation: 1
  - name: Quentin Glorieux
    orcid: 0000-0003-0903-0233
    affiliation: 1
affiliations:
 - name: Laboratoire Kastler Brossel, Sorbonne University, CNRS, ENS-PSL University, Collège de France; 4 Place Jussieu, 75005 Paris, France
   index: 1
date: 15 March 2024
bibliography: paper.bib
---

# Summary

Off-axis interferometry is a powerful technique that allows full-field retrieval from interferograms [@lieblingComplexwaveRetrievalSingle2004]
[@verrierOffaxisDigitalHologram2011].
Based on the deconvolution of the inteferograms using Fourier transforms, it allows live monitoring of optical fields.
It comprises of three utilities `contrast`, `velocity` and `SLM`:

- `contrast` is focused on the retrieval of the phase.
- `velocity` is focused on the processing of the complex field.
- `SLM` provides a performant, platform independant window-based tool to control spatial light modulators such as liquid crystal spatial light modulators (LCOS SLM) or digital micro-mirror devices (DMD).

# Statement of need

Phase retrieval is a critical topic in all optics experiment.
It is often challenging since cameras can only access intensity information.
The solution to this problem is spatially heterodyning a target signal with a reference signal using the interference signal to recover the phase.
This is the so-called off-axis interferometry technique [@verrierOffaxisDigitalHologram2011].
It allows a singe shot, high resolution retrieval of the full complex optical field.

`PhaseUtils` harnesses the power of modern FFT algorithms to deliver performance focused tools to retrieve and process the phase information of optical fields with its utilities `contrast` and `velocity`.
This allows to compute a large number of observables relevant in the context of quantum fluids of light [@aladjidiFullOpticalControl2023] [@glorieuxHotAtomicVapors2023] [@bakerrasooliTurbulentDynamicsTwodimensional2023].
`velocity` implements all of the observables introduced in [@bradleyEnergySpectraVortex2012] and implemented in the `Julia` package [`QuantumFluidSpectra.jl`](https://github.com/AshtonSBradley/QuantumFluidSpectra.jl) [@PhysRevA.106.043322], and extends it by providing a fast tree-based implementation of the vortex clustering algorithm.

![Example of the vortex detection and clustering algorithm. Positively charged vortices are in red, negatively charged vortices are blue and dipoles are in green. The background image is the incompressible velocity in which vortices can be seen as bright peaks.\label{fig:clusters}](../assets/clusters.png)

It also provide tools with `SLM` to control the optical fields using spatial light modulators, implementing holography techniques such as [@bolducExactSolutionSimultaneous2013].
This utility was inspired by [@sebastien_m_popoff_2017_293042] and implements the same basic functionalities, using a faster backend (`opencv`).
Since it functions by instantiating a graphical window, it can be used for any spatial light modulator that is recognized as a screen.
It also extends the simple control functionality with all of the relevant functions to
generate arbitrary states of light using these modulators.
These functions use JIT compilation with `numba` for optimal performance to allow for fast live control.

# Acknowledgements

We acknowledge the extremely meaningful conversations we had with Riccardo Panico and Ashton Bradley.
We acknowledge contributions from Kevin Falque and constructive feedback from Clara Piekarski and Quentin Schibler.

# Authors contribution

T.A wrote the original code and is the main maintainer, M.B has contributed on the vortex detection and classification routines. Q.G supervised the project.

# References