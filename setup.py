"""Init module for the PhaseUtils package."""

from setuptools import setup

setup(
    name="PhaseUtils",
    version="1.0.0",
    description="A package containing all the relevant utilities to process phase data of optical fields.",
    url="https://github.com/Quantum-Optics-LKB/PhaseUtils",
    author="Tangui Aladjidi",
    author_email="tangui.aladjidi@lkb.upmc.fr",
    license="MIT",
    packages=["PhaseUtils"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "pyfftw",
        "numba",
        "scikit-image",
        "cupy",
        "networkx",
        "numbalsoda",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Environment :: GPU",
        "Environment :: GPU :: NVIDIA CUDA :: 10",
        "Environment :: GPU :: NVIDIA CUDA :: 11",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
    ],
)
