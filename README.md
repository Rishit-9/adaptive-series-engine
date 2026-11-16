# Adaptive Function Approximation Engine (AFAE)

**A Hybrid Neuro-Symbolic Series Expansion Optimization Framework**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow CPU 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00.svg)](https://www.tensorflow.org/)
[![SymPy 1.12+](https://img.shields.io/badge/SymPy-1.12+-green.svg)](https://www.sympy.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-009688.svg)](https://fastapi.tiangolo.com/)

## Overview

The **Adaptive Function Approximation Engine (AFAE)** is a scientific computing library and REST API that bridges deterministic symbolic mathematics with deep learning optimization. Classical series expansions—such as Taylor, Fourier, and Chebyshev polynomials—frequently suffer from localized radius of convergence limits, the Gibbs phenomenon at jump discontinuities, and Runge's phenomenon at interval boundaries. 

Instead of training black-box neural networks from random initialization, AFAE uses **SymPy** and **SciPy** to compute exact analytical base coefficients (<i>C</i><sub>base</sub>) and hands them off to a custom **TensorFlow** optimization layer. The neural layer refines the coefficients (<i>C</i><sub>refined</sub> = <i>C</i><sub>base</sub> + Δ<i>C</i>) against a multi-objective loss function that penalizes peak overshoot (<i>L</i><sub>∞</sub>) and total variation oscillation.

### 🌐 Live API Demo
The engine is deployed as a live interactive API. You can test the endpoints and view the auto-generated Swagger UI here:
**[https://adaptive-series-engine.onrender.com/docs](https://adaptive-series-engine.onrender.com/docs)**

## Key Features

* **RESTful API Deployment:** Fully interactive web interface powered by FastAPI for instant function approximation and basis routing.
* **Symbolic Base Generation:** Exact N-th order symbolic differentiation for Taylor series, numerical quadrature for Fourier harmonics, and orthogonal Chebyshev node interpolation.
* **Neuro-Symbolic Weight Anchoring:** Custom Keras `HybridSeriesLayer` initializes network weights directly from symbolic coefficients, drastically reducing convergence time.
* **Oscillation-Damped Custom Loss:** Combines Mean Squared Error (MSE), <i>L</i><sub>∞</sub> maximum overshoot suppression, Total Variation (TV) derivative smoothing, and <i>L</i><sub>2</sub> anchor regularization.

## Repository Structure

```text
adaptive-series-engine/
├── src/
│   ├── __init__.py
│   ├── symbolic_engine.py      # Symbolic & numerical basis expansion generators
│   ├── router.py               # Adaptive routing logic for basis selection
│   └── neural_optimizer.py     # Custom TensorFlow layer and hybrid loss optimizer
├── main.py                     # FastAPI application wrapper
├── prototype.py                # Single-variable benchmark & plotting suite
├── requirements.txt            # Package dependencies
└── README.md                   # Documentation
```

## Installation

Clone the repository and install the required dependencies (memory-optimized for CPU environments):

```bash
git clone https://github.com/your-username/adaptive-series-engine.git
cd adaptive-series-engine
pip install -r requirements.txt
```

## Usage

### 1. Running the API Locally
To launch the FastAPI server on your local machine:

```bash
uvicorn main:app --reload
```
Once running, navigate to `http://127.0.0.1:8000/docs` in your browser to interact with the engine.

### 2. API Request Example
You can send a POST request to the `/api/v1/optimize/taylor` endpoint to optimize an expression:

```json
POST /api/v1/optimize/taylor
{
  "expression": "cos(x) * exp(-0.1 * x**2)",
  "x_min": -3.14159,
  "x_max": 3.14159,
  "num_points": 400,
  "n_terms": 7,
  "epochs": 300
}
```

### 3. Using the Python Library
Run the local benchmarking suites to visualize Taylor boundary stabilization and Fourier Gibbs suppression:

```bash
python prototype.py
python comprehensive_validation.py
```
Instead of training black-box neural networks from random initialization, AFAE uses **SymPy** and **SciPy** to compute exact analytical base coefficients (<i>C</i><sub>base</sub>) and hands them off to a custom **TensorFlow** optimization layer. The neural layer refines the coefficients (<i>C</i><sub>refined</sub> = <i>C</i><sub>base</sub> + Δ<i>C</i>) against a multi-objective loss function that penalizes peak overshoot (<i>L</i><sub>∞</sub>) and total variation oscillation.

## Key Features

* **Symbolic Base Generation:** Exact N-th order symbolic differentiation for Taylor series, numerical quadrature for Fourier harmonics, and orthogonal Chebyshev node interpolation.
* **Neuro-Symbolic Weight Anchoring:** Custom Keras `HybridSeriesLayer` initializes network weights directly from symbolic coefficients, drastically reducing convergence time.
* **Oscillation-Damped Custom Loss:** Combines Mean Squared Error (MSE), <i>L</i><sub>∞</sub> maximum overshoot suppression, Total Variation (TV) derivative smoothing, and <i>L</i><sub>2</sub> anchor regularization.
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
```

## Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/rishit-9/adaptive-series-engine.git](https://github.com/rishit-9/adaptive-series-engine.git)
cd adaptive-series-engine
pip install -r requirements.txt
```

## Quick Start

Run the single-variable benchmark suite to evaluate Taylor boundary stabilization and Fourier Gibbs suppression:

```bash
python prototype.py
```

### Example Usage in Python

```python
import numpy as np
import sympy as sp
from src.symbolic_engine import SymbolicExpansionEngine
from src.neural_optimizer import NeuralCoefficientOptimizer

# 1. Initialize Symbolic Engine
engine = SymbolicExpansionEngine()
x_sym = engine.x
target_expr = sp.cos(x_sym) * sp.exp(-0.1 * x_sym**2)
target_fn = sp.lambdify(x_sym, target_expr, "numpy")

x_vals = np.linspace(-3.2, 3.2, 400, dtype=np.float32)
y_true = target_fn(x_vals).astype(np.float32)

# 2. Extract Symbolic Base Coefficients (C_base)
c_base, basis_matrix = engine.taylor_expansion(target_expr, x_vals, center=0.0, n_terms=7)

# 3. Refine Coefficients via Neural Optimizer (C_refined = C_base + delta_C)
optimizer = NeuralCoefficientOptimizer(c_base, learning_rate=0.005, alpha_peak=0.1, beta_tv=0.02)
c_refined, y_hybrid = optimizer.fit(basis_matrix, y_true, epochs=600)
```