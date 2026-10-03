# Adaptive Function Approximation Engine (AFAE)

**A Hybrid Neuro-Symbolic Series Expansion Optimization Framework**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00.svg)](https://www.tensorflow.org/)
[![SymPy 1.12+](https://img.shields.io/badge/SymPy-1.12+-green.svg)](https://www.sympy.org/)

## Overview

The **Adaptive Function Approximation Engine (AFAE)** is a scientific computing library that bridges deterministic symbolic mathematics with deep learning optimization. Classical series expansions—such as Taylor, Fourier, and Chebyshev polynomials—frequently suffer from localized radius of convergence limits, the Gibbs phenomenon at jump discontinuities, and Runge's phenomenon at interval boundaries. 

Instead of training black-box neural networks from random initialization, AFAE uses **SymPy** and **SciPy** to compute exact analytical base coefficients ($C_{base}$) and hands them off to a custom **TensorFlow** optimization layer. The neural layer refines the coefficients ($C_{refined} = C_{base} + \Delta C$) against a multi-objective loss function that penalizes peak overshoot ($L_\infty$) and total variation oscillation.

## Key Features

* **Symbolic Base Generation:** Exact N-th order symbolic differentiation for Taylor series, numerical quadrature for Fourier harmonics, and orthogonal Chebyshev node interpolation.
* **Neuro-Symbolic Weight Anchoring:** Custom Keras `HybridSeriesLayer` initializes network weights directly from symbolic coefficients, drastically reducing convergence time.
* **Oscillation-Damped Custom Loss:** Combines Mean Squared Error (MSE), $L_\infty$ maximum overshoot suppression, Total Variation (TV) derivative smoothing, and $L_2$ anchor regularization.
* **Real-Time Convergence Diagnostics:** Built-in Matplotlib visualization comparing static analytical baselines against hybrid refined curves.

## Repository Structure

```text
adaptive-series-engine/
├── src/
│   ├── __init__.py
│   ├── symbolic_engine.py      # Symbolic & numerical basis expansion generators
│   └── neural_optimizer.py     # Custom TensorFlow layer and hybrid loss optimizer
├── prototype.py                # Single-variable benchmark & plotting suite
├── requirements.txt            # Package dependencies
└── README.md                   # Documentation
