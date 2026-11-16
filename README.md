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

* **Intelligent Auto-Routing:** Analyzes function variance, smoothness, and periodicity to dynamically select the optimal mathematical basis (Taylor, Fourier, or Chebyshev).
* **RESTful API Deployment:** Fully interactive web interface powered by FastAPI for instant function approximation.
* **Symbolic Base Generation:** Exact N-th order symbolic differentiation, numerical quadrature for Fourier harmonics, and orthogonal Chebyshev node interpolation.
* **Neuro-Symbolic Weight Anchoring:** Custom Keras `HybridSeriesLayer` initializes network weights directly from symbolic coefficients, drastically reducing convergence time.
* **Oscillation-Damped Custom Loss:** Combines Mean Squared Error (MSE), <i>L</i><sub>∞</sub> maximum overshoot suppression, Total Variation (TV) derivative smoothing, and <i>L</i><sub>2</sub> anchor regularization.

## Repository Structure

```text
adaptive-series-engine/
├── src/
│   ├── __init__.py            
│   ├── neural_optimizer.py     
│   ├── router.py               
│   └── symbolic_engine.py      
├── Dockerfile               
├── README.md                   
├── comprehensive_validation.py 
├── main.py                    
├── prototype.py                
├── pyproject.toml              
├── requirements.txt            
└── runge_benchmark.py          
```

## Installation

AFAE is packaged as a deployable Python library. You can install it locally for development and testing:

```bash
git clone https://github.com/your-username/adaptive-series-engine.git
cd adaptive-series-engine
pip install -e .
```

## Usage

### 1. Running the Web API Locally
To launch the FastAPI server on your own machine:
```bash
uvicorn main:app --reload
```
Navigate to `http://127.0.0.1:8000/docs` in your browser to view the local Swagger UI.

### 2. API Request Example (Auto-Routing)
You can send a POST request to the `/api/v1/optimize/auto` endpoint. The engine will evaluate the expression, choose the best basis automatically, and apply neural refinement:

```json
POST /api/v1/optimize/auto
{
  "expression": "cos(x) * exp(-0.1 * x**2)",
  "x_min": -3.14159,
  "x_max": 3.14159,
  "num_points": 400,
  "n_terms": 7,
  "epochs": 300
}
```

*Note: Manual override endpoints (`/optimize/taylor`, `/optimize/fourier`, `/optimize/chebyshev`) are also available if strict basis enforcement is required.*

### 3. Using the Python Library Locally
Run the built-in benchmarking suites to visualize Taylor boundary stabilization, Fourier Gibbs suppression, and Runge's phenomenon mitigation:

```bash
python prototype.py
python runge_benchmark.py
python comprehensive_validation.py
```