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
It comprises of two packages `contrast` and `velocity`.
`contrast` is focused on the retrieval of the phase.
`velocity` is focused on the processing of the complex field.

# Statement of need

`PhaseUtils` harnesses the power of modern FFT algorithms to deliver performance focused tools to retrieve and process the phase information of optical fields.
This allows to compute a large number of observables relevant in the context of quantum fluids of light [@aladjidiFullOpticalControl2023] [@glorieuxHotAtomicVapors2023].

# Acknowledgements

We acknowledge contributions from Gilles, Maïkeul and Bébert le camembert.

# Authors contribution

T.A wrote the original code and is the main maintainer, Myrann Baker-Rasooli has contributed on the vortex detection and classification routines.

# References