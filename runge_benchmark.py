import numpy as np
import matplotlib.pyplot as plt
from src.symbolic_engine import SymbolicExpansionEngine
from src.neural_optimizer import NeuralCoefficientOptimizer
from src.router import AdaptiveBasisRouter

def run_integration_benchmark():
    print("--- Milestone 3: Integration & Runge's Phenomenon Benchmark ---")
    
    # Runge's function: f(x) = 1 / (1 + 25x^2)
    x_vals = np.linspace(-1, 1, 100, dtype=np.float32)
    y_true = 1.0 / (1.0 + 25.0 * x_vals**2)

    # 1. Adaptive Routing
    router = AdaptiveBasisRouter()
    selected_basis = router.route(x_vals, y_true)
    print(f"[Router] Selected Basis: {selected_basis.upper()}")

    # 2. Symbolic Generation
    engine = SymbolicExpansionEngine()
    if selected_basis == "chebyshev":
        c_base, phi_matrix = engine.chebyshev_expansion(lambda x: 1/(1+25*x**2), x_vals, n_terms=11)
    
    y_static = phi_matrix @ c_base

    # 3. Neural Optimization
    optimizer = NeuralCoefficientOptimizer(c_base, learning_rate=0.01, alpha_peak=0.3)
    c_refined, y_hybrid = optimizer.fit(phi_matrix, y_true, epochs=400)

    print(f"Static MSE (Runge Divergence): {np.mean((y_true - y_static)**2):.4f}")
    print(f"Hybrid MSE (Mitigated): {np.mean((y_true - y_hybrid)**2):.4f}")

if __name__ == "__main__":
    run_integration_benchmark()